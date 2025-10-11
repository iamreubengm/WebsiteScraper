import os
import string
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import openai
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ============================================================================
# 1. TEXT NORMALIZATION UTILITIES
# ============================================================================

class TextNormalizer:
    
    ARTICLES = {'a', 'an', 'the'}
    
    @staticmethod
    def normalize_answer(s):        
        def remove_articles(text):
            return ' '.join([word for word in text.split() if word.lower() not in TextNormalizer.ARTICLES])
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return TextNormalizer.normalize_answer(s).split()

# ============================================================================
# 2. CORE METRICS (EM, F1, Precision, Recall)
# ============================================================================

class CoreMetrics:
    
    @staticmethod
    def exact_match(prediction, ground_truth):
        return float(TextNormalizer.normalize_answer(prediction) == 
                    TextNormalizer.normalize_answer(ground_truth))
    
    @staticmethod
    def f1_score(prediction, ground_truth):
        pred_tokens = TextNormalizer.get_tokens(prediction)
        truth_tokens = TextNormalizer.get_tokens(ground_truth)
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    @staticmethod
    def precision(prediction, ground_truth):
        pred_tokens = TextNormalizer.get_tokens(prediction)
        truth_tokens = TextNormalizer.get_tokens(ground_truth)
        
        if len(pred_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        return num_same / len(pred_tokens)
    
    @staticmethod
    def recall(prediction, ground_truth):
        pred_tokens = TextNormalizer.get_tokens(prediction)
        truth_tokens = TextNormalizer.get_tokens(ground_truth)
        
        if len(truth_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        return num_same / len(truth_tokens)


# ============================================================================
# 3. ROUGE METRICS
# ============================================================================

class RougeMetrics:
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        
    
    def rouge2_score(self, prediction, ground_truth):
        if not self.scorer:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_norm = TextNormalizer.normalize_answer(prediction)
        truth_norm = TextNormalizer.normalize_answer(ground_truth)
        
        scores = self.scorer.score(truth_norm, pred_norm)
        rouge2 = scores['rouge2']
        
        return {
            'precision': rouge2.precision,
            'recall': rouge2.recall,
            'f1': rouge2.fmeasure
        }


# ============================================================================
# 4. BERT SCORE METRICS
# ============================================================================

class BertScoreMetrics:
    
    @staticmethod
    def compute_bert_score(predictions, 
                          references,
                          lang = 'en',
                          model_type = 'microsoft/deberta-xlarge-mnli'):
        """
        Compute BERTScore for batch of predictions
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
            lang: Language code
            model_type: BERT model to use for scoring
        """
                    
        
        P, R, F1 = bert_score(predictions, references, 
                             lang=lang, model_type=model_type, verbose=False)
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }


# ============================================================================
# 5. LLM AS A JUDGE
# ============================================================================

class JudgementCriteria(Enum):
    """Criteria for LLM evaluation"""
    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    RETRIEVAL_RELEVANCE = "retrieval_relevance"


@dataclass
class LLMJudgement:    
    correctness: float
    groundedness: float
    relevance: float
    retrieval_relevance: float
    reasoning: Dict[str, str]


class LLMJudge:
    """LLM-based evaluation using OpenAI API"""
    
    PROMPT_TEMPLATES = {
        JudgementCriteria.CORRECTNESS: """
You are an expert evaluator. Rate the CORRECTNESS of the predicted answer compared to the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Rate from 0-10 where:
- 0: Completely incorrect or irrelevant
- 5: Partially correct but missing key information
- 10: Perfectly correct and complete

Provide your rating and brief reasoning.
Output format:
SCORE: [0-10]
REASONING: [Your explanation]
""",
        
        JudgementCriteria.GROUNDEDNESS: """
You are an expert evaluator. Rate the GROUNDEDNESS of the predicted answer based on the retrieved context.

Question: {question}
Retrieved Context: {context}
Predicted Answer: {prediction}

Rate from 0-10 where:
- 0: Answer contains unsupported claims or hallucinations
- 5: Partially grounded with some unsupported details
- 10: Fully grounded in the provided context

Provide your rating and brief reasoning.
Output format:
SCORE: [0-10]
REASONING: [Your explanation]
""",
        
        JudgementCriteria.RELEVANCE: """
You are an expert evaluator. Rate the RELEVANCE of the predicted answer to the question.

Question: {question}
Predicted Answer: {prediction}

Rate from 0-10 where:
- 0: Completely irrelevant to the question
- 5: Somewhat relevant but includes off-topic information
- 10: Perfectly relevant and directly addresses the question

Provide your rating and brief reasoning.
Output format:
SCORE: [0-10]
REASONING: [Your explanation]
""",
        
        JudgementCriteria.RETRIEVAL_RELEVANCE: """
You are an expert evaluator. Rate the RETRIEVAL RELEVANCE of the retrieved context to the question.

Question: {question}
Retrieved Context: {context}

Rate from 0-10 where:
- 0: Context is completely irrelevant to the question
- 5: Context is somewhat related but missing key information
- 10: Context is highly relevant and contains necessary information

Provide your rating and brief reasoning.
Output format:
SCORE: [0-10]
REASONING: [Your explanation]
"""
    }
    
    def __init__(self, api_key, base_url= None, model = "gpt-4.1-mini"):    
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def _parse_llm_response(self, response):
        try:
            lines = response.strip().split('\n')
            score_line = [l for l in lines if l.startswith('SCORE:')][0]
            reasoning_line = [l for l in lines if l.startswith('REASONING:')][0]
            
            score = float(score_line.split(':')[1].strip())
            reasoning = reasoning_line.split(':', 1)[1].strip()
            
            return score / 10.0, reasoning
        except:
            return 0.5, "Failed to parse LLM response"
    
    def evaluate(self, 
                question,
                prediction,
                ground_truth,
                context = None) -> LLMJudgement:
        """
        Evaluate prediction using LLM across all criteria
        
        Args:
            question: The question being answered
            prediction: The predicted answer
            ground_truth: The ground truth answer
            context: Retrieved context (optional, needed for groundedness/retrieval relevance)
        """
        results = {}
        reasoning = {}
        
        # Correctness
        prompt = self.PROMPT_TEMPLATES[JudgementCriteria.CORRECTNESS].format(
            question=question, ground_truth=ground_truth, prediction=prediction
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score, reason = self._parse_llm_response(response.choices[0].message.content)
        results['correctness'] = score
        reasoning['correctness'] = reason
        
        # Relevance
        prompt = self.PROMPT_TEMPLATES[JudgementCriteria.RELEVANCE].format(
            question=question, prediction=prediction
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score, reason = self._parse_llm_response(response.choices[0].message.content)
        results['relevance'] = score
        reasoning['relevance'] = reason
        
        # Groundedness and Retrieval Relevance (if context provided)
        if context:
            # Groundedness
            prompt = self.PROMPT_TEMPLATES[JudgementCriteria.GROUNDEDNESS].format(
                question=question, context=context, prediction=prediction
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score, reason = self._parse_llm_response(response.choices[0].message.content)
            results['groundedness'] = score
            reasoning['groundedness'] = reason
            
            # Retrieval Relevance
            prompt = self.PROMPT_TEMPLATES[JudgementCriteria.RETRIEVAL_RELEVANCE].format(
                question=question, context=context
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score, reason = self._parse_llm_response(response.choices[0].message.content)
            results['retrieval_relevance'] = score
            reasoning['retrieval_relevance'] = reason
        else:
            results['groundedness'] = 0.0
            results['retrieval_relevance'] = 0.0
            reasoning['groundedness'] = "No context provided"
            reasoning['retrieval_relevance'] = "No context provided"
        
        return LLMJudgement(
            correctness=results['correctness'],
            groundedness=results['groundedness'],
            relevance=results['relevance'],
            retrieval_relevance=results['retrieval_relevance'],
            reasoning=reasoning
        )


# ============================================================================
# 6. EVALUATION RESULT STRUCTURES
# ============================================================================

@dataclass
class EvaluationResult:
    """Single evaluation result for one prediction"""
    question_id: str
    question: str
    prediction: str
    ground_truth: str
    context: Optional[str]
    
    # Core metrics
    exact_match: float
    f1: float
    precision: float
    recall: float
    
    # ROUGE-2
    rouge2_precision: float
    rouge2_recall: float
    rouge2_f1: float
    
    # BERTScore
    bert_precision: float
    bert_recall: float
    bert_f1: float
    
    # LLM Judge
    llm_correctness: float
    llm_groundedness: float
    llm_relevance: float
    llm_retrieval_relevance: float
    llm_reasoning: Optional[Dict[str, str]] = None
    
    # Metadata for analysis
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AggregatedResults:
    """Aggregated results across all predictions"""
    total_samples: int
    
    # Core metrics (averages)
    exact_match: float
    f1: float
    precision: float
    recall: float
    
    # ROUGE-2 (averages)
    rouge2_precision: float
    rouge2_recall: float
    rouge2_f1: float
    
    # BERTScore (averages)
    bert_precision: float
    bert_recall: float
    bert_f1: float
    
    # LLM Judge (averages)
    llm_correctness: float
    llm_groundedness: float
    llm_relevance: float
    llm_retrieval_relevance: float
    
    # Distribution stats
    em_distribution: Dict[str, int]  # Count of EM scores
    f1_ranges: Dict[str, int]  # Count in F1 ranges


# ============================================================================
# 7. MAIN EVALUATOR CLASS
# ============================================================================

class Evaluator:
    """Main evaluator orchestrating all metrics"""
    
    def __init__(self, 
                 use_llm_judge: bool = False,
                 llm_api_key= None,
                 llm_base_url= None,
                 llm_model = "gpt-4.1-mini"):
        """
        Initialize evaluator
        
        Args:
            use_llm_judge: Whether to use LLM as judge
            llm_api_key: OpenAI API key (required if use_llm_judge=True)
            llm_base_url: Custom base URL for OpenAI API
            llm_model: Model to use for LLM judge
        """
        self.core_metrics = CoreMetrics()
        self.rouge_metrics = RougeMetrics()
        self.bert_metrics = BertScoreMetrics()
        
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            if not llm_api_key:
                raise ValueError("llm_api_key required when use_llm_judge=True")
            self.llm_judge = LLMJudge(llm_api_key, llm_base_url, llm_model)
        else:
            self.llm_judge = None
    
    def evaluate_single(self,
                       question_id,
                       question,
                       prediction,
                       ground_truth,
                       context= None,
                       metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single prediction
        
        Args:
            question_id: Unique identifier for the question
            question: The question text
            prediction: Predicted answer
            ground_truth: Ground truth answer
            context: Retrieved context (optional)
            metadata: Additional metadata for analysis
        """
        # Core metrics
        em = self.core_metrics.exact_match(prediction, ground_truth)
        f1 = self.core_metrics.f1_score(prediction, ground_truth)
        precision = self.core_metrics.precision(prediction, ground_truth)
        recall = self.core_metrics.recall(prediction, ground_truth)
        
        # ROUGE-2
        rouge2 = self.rouge_metrics.rouge2_score(prediction, ground_truth)
        
        # BERTScore (computed in batch, placeholder here)
        bert_p, bert_r, bert_f1 = 0.0, 0.0, 0.0
        
        # LLM Judge
        if self.use_llm_judge and self.llm_judge:
            llm_result = self.llm_judge.evaluate(question, prediction, ground_truth, context)
            llm_correctness = llm_result.correctness
            llm_groundedness = llm_result.groundedness
            llm_relevance = llm_result.relevance
            llm_retrieval_relevance = llm_result.retrieval_relevance
            llm_reasoning = llm_result.reasoning
        else:
            llm_correctness = 0.0
            llm_groundedness = 0.0
            llm_relevance = 0.0
            llm_retrieval_relevance = 0.0
            llm_reasoning = None
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            context=context,
            exact_match=em,
            f1=f1,
            precision=precision,
            recall=recall,
            rouge2_precision=rouge2['precision'],
            rouge2_recall=rouge2['recall'],
            rouge2_f1=rouge2['f1'],
            bert_precision=bert_p,
            bert_recall=bert_r,
            bert_f1=bert_f1,
            llm_correctness=llm_correctness,
            llm_groundedness=llm_groundedness,
            llm_relevance=llm_relevance,
            llm_retrieval_relevance=llm_retrieval_relevance,
            llm_reasoning=llm_reasoning,
            metadata=metadata
        )
    
    def evaluate_batch(self,
                      questions: List[Dict[str, Any]],
                      compute_bert: bool = True) -> List[EvaluationResult]:
        """
        Evaluate a batch of predictions
        
        Args:
            questions: List of dicts with keys: question_id, question, prediction, 
                      ground_truth, context (optional), metadata (optional)
            compute_bert: Whether to compute BERTScore
        
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        # compute all metrics except BERTScore
        for q in tqdm(questions, desc="Scoring (core + ROUGE)", unit="ex"):        
            result = self.evaluate_single(
                question_id=q['question_id'],
                question=q['question'],
                prediction=q['prediction'],
                ground_truth=q['ground_truth'],
                context=q.get('context'),
                metadata=q.get('metadata')
            )
            results.append(result)
        
        # compute BERTScore in batch if requested
        if compute_bert:
            logging.info("Computing BERTScore (batch)...")
            predictions = [q['prediction'] for q in questions]
            ground_truths = [q['ground_truth'] for q in questions]
            
            bert_scores = self.bert_metrics.compute_bert_score(predictions, ground_truths)
            
            for i, result in enumerate(results):
                result.bert_precision = bert_scores['precision'][i]
                result.bert_recall = bert_scores['recall'][i]
                result.bert_f1 = bert_scores['f1'][i]
        
        return results
    
    def aggregate_results(self, results: List[EvaluationResult]) -> AggregatedResults:
        """Aggregate evaluation results"""
        n = len(results)
        
        if n == 0:
            return AggregatedResults(
                total_samples=0,
                exact_match=0.0, f1=0.0, precision=0.0, recall=0.0,
                rouge2_precision=0.0, rouge2_recall=0.0, rouge2_f1=0.0,
                bert_precision=0.0, bert_recall=0.0, bert_f1=0.0,
                llm_correctness=0.0, llm_groundedness=0.0,
                llm_relevance=0.0, llm_retrieval_relevance=0.0,
                em_distribution={}, f1_ranges={}
            )
        
        # Calculate averages
        avg_em = np.mean([r.exact_match for r in results])
        avg_f1 = np.mean([r.f1 for r in results])
        avg_precision = np.mean([r.precision for r in results])
        avg_recall = np.mean([r.recall for r in results])
        
        avg_rouge2_p = np.mean([r.rouge2_precision for r in results])
        avg_rouge2_r = np.mean([r.rouge2_recall for r in results])
        avg_rouge2_f1 = np.mean([r.rouge2_f1 for r in results])
        
        avg_bert_p = np.mean([r.bert_precision for r in results])
        avg_bert_r = np.mean([r.bert_recall for r in results])
        avg_bert_f1 = np.mean([r.bert_f1 for r in results])
        
        avg_llm_correctness = np.mean([r.llm_correctness for r in results])
        avg_llm_groundedness = np.mean([r.llm_groundedness for r in results])
        avg_llm_relevance = np.mean([r.llm_relevance for r in results])
        avg_llm_retrieval = np.mean([r.llm_retrieval_relevance for r in results])
        
        # EM distribution
        em_dist = {
            'exact_matches': sum(1 for r in results if r.exact_match == 1.0),
            'no_matches': sum(1 for r in results if r.exact_match == 0.0)
        }
        
        # F1 ranges
        f1_ranges = {
            '0.0-0.2': sum(1 for r in results if 0.0 <= r.f1 < 0.2),
            '0.2-0.4': sum(1 for r in results if 0.2 <= r.f1 < 0.4),
            '0.4-0.6': sum(1 for r in results if 0.4 <= r.f1 < 0.6),
            '0.6-0.8': sum(1 for r in results if 0.6 <= r.f1 < 0.8),
            '0.8-1.0': sum(1 for r in results if 0.8 <= r.f1 <= 1.0)
        }
        
        return AggregatedResults(
            total_samples=n,
            exact_match=avg_em,
            f1=avg_f1,
            precision=avg_precision,
            recall=avg_recall,
            rouge2_precision=avg_rouge2_p,
            rouge2_recall=avg_rouge2_r,
            rouge2_f1=avg_rouge2_f1,
            bert_precision=avg_bert_p,
            bert_recall=avg_bert_r,
            bert_f1=avg_bert_f1,
            llm_correctness=avg_llm_correctness,
            llm_groundedness=avg_llm_groundedness,
            llm_relevance=avg_llm_relevance,
            llm_retrieval_relevance=avg_llm_retrieval,
            em_distribution=em_dist,
            f1_ranges=f1_ranges
        )


# ============================================================================
# 8. ANALYSIS UTILITIES
# ============================================================================

def strip_wrappers(raw_line):    
    line = raw_line.strip()
    if not (line.startswith('{') and line.endswith('}')):        
        first = line.find('{')
        last = line.rfind('}')
        if first != -1 and last != -1 and last > first:
            line = line[first:last + 1]
    return line

def load_jsonl(path):    
    rows = []
    logging.info(f"Loading JSONL from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, raw in enumerate(tqdm(lines, desc="Reading JSONL", unit="line"), start=1):
            if not raw.strip():
                continue
            try:
                cleaned = strip_wrappers(raw)
                obj = json.loads(cleaned)
                if not isinstance(obj, dict):
                    raise ValueError("Line does not contain a JSON object.")
                rows.append(obj)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {i}: {e}\nRaw: {raw[:200]}") from e
    logging.info(f"Loaded {len(rows)} records.")
    return rows

def to_evaluator_batch(jsonl_rows):
    batch = []
    for i, r in enumerate(tqdm(jsonl_rows, desc="Mapping to evaluator schema", unit="rec"), start=1):
        try:
            qid = r.get("question_id")
            question = r.get("query")
            prediction = r.get("generated_answer")
            ground_truth = r.get("reference_answer")
            context = r.get("context_used")

            if question is None or prediction is None or ground_truth is None:
                raise KeyError("Missing one of required keys: 'query', 'generated_answer', 'reference_answer'.")

            batch.append({
                "question_id": qid,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "context": context,
                "metadata": None
            })
        except Exception as e:
            raise ValueError(f"Row {i} is malformed for evaluator mapping: {e}\nRow: {r}") from e
    logging.info(f"Prepared batch of {len(batch)} examples.")
    return batch

def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate JSONL Q&A predictions.")
    parser.add_argument("--file", "-f", required=True, help="Path to input .jsonl file")
    parser.add_argument("--out-dir", default=None, help="Directory to write results (per_sample.jsonl, aggregate.json)")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERTScore computation")
    parser.add_argument("--use-llm-judge", action="store_true", help="Enable LLM judge (requires --llm-api-key)")
    parser.add_argument("--llm-api-key", default=None, help="API key for LLM judge")
    parser.add_argument("--llm-base-url", default=None, help="Custom base URL for LLM judge")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="LLM judge model name")
    args = parser.parse_args()

    rows = load_jsonl(args.file)
    batch = to_evaluator_batch(rows)
    
    if args.use_llm_judge and not args.llm_api_key:
        raise SystemExit("ERROR: --use-llm-judge requires --llm-api-key")
    evaluator = Evaluator(
        use_llm_judge=args.use_llm_judge,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model
    )
    
    logging.info("Starting evaluation...")
    results = evaluator.evaluate_batch(batch, compute_bert=(not args.no_bert))
    logging.info("Evaluation complete. Aggregating...")
    agg = evaluator.aggregate_results(results)
    
    print("\n=== Per-sample results ===")
    per_sample_rows = []
    for r in results:
        rd = asdict(r)   
        summary = {
            "question_id": rd.get("question_id"),
            "question": rd.get("question"),
            "prediction": rd.get("prediction"),
            "ground_truth": rd.get("ground_truth"),                    
            "EM": rd.get("exact_match"),
            "F1": rd.get("f1"),
            "Precision": rd.get("precision"),
            "Recall": rd.get("recall"),
            "ROUGE-2 F1": rd.get("rouge2_f1"),
            "BERT F1": rd.get("bert_f1"),
            "llm_correctness": rd.get("llm_correctness"),
            "llm_groundedness": rd.get("llm_groundedness"),
            "llm_relevance": rd.get("llm_relevance"),
            "llm_retrieval_relevance": rd.get("llm_retrieval_relevance"),
        }
        per_sample_rows.append(summary)        
        print(json.dumps(summary, ensure_ascii=False))

    # Print aggregate
    print("\n=== Aggregated results ===")
    agg_obj = asdict(agg)
    print(json.dumps(agg_obj, indent=2, ensure_ascii=False))

    if args.out_dir:
        per_sample_path = os.path.join(args.out_dir, "per_sample.jsonl")
        aggregate_path = os.path.join(args.out_dir, "aggregate.json")
        save_jsonl(per_sample_path, per_sample_rows)
        save_json(aggregate_path, agg_obj)
        logging.info(f"Wrote per-sample to {per_sample_path}")
        logging.info(f"Wrote aggregate to {aggregate_path}")
    logging.info("Done.")

if __name__ == "__main__":
    main()