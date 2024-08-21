import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import argparse
from tabulate import tabulate

import os
import ssl

# Set the NLTK data path to a directory where you have write permissions
nltk.data.path.append(os.path.expanduser("~/nltk_data"))

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
    
# Initialize SentenceTransformer model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Rouge for text similarity
rouge = Rouge()

SERVER_URL = "http://localhost:8001"

def get_response(query, provider="ollama", model="llama3.1"):
    """Send a query to the Q&A system and get the response."""
    payload = {
        "provider": provider,
        "model": model,
        "prompt": query,
        "max_tokens": 200
    }
    try:
        response = requests.post(f"{SERVER_URL}/generate", json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        print("Full response:", response_data)
        
        if 'answer' in response_data:
            return response_data['answer']
        elif 'response' in response_data:
            return response_data['response']
        elif isinstance(response_data, dict) and len(response_data) > 0:
            return next(iter(response_data.values()))
        else:
            return f"Unexpected response structure: {response_data}"
    except requests.RequestException as e:
        return f"Error connecting to server: {str(e)}"


def semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts."""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score for the response."""
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    return sentence_bleu([reference_tokens], candidate_tokens)

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores for the response."""
    scores = rouge.get_scores(candidate, reference)
    return scores[0]

def evaluate_response(query, response, reference=None):
    """Evaluate the response based on multiple metrics."""
    evaluation = {}
    
    # Semantic Similarity
    evaluation['semantic_similarity'] = semantic_similarity(query, response)
    
    # Response Length
    evaluation['response_length'] = len(response.split())
    
    if reference:
        # Reference-based metrics
        evaluation['ref_similarity'] = semantic_similarity(reference, response)
        evaluation['bleu_score'] = calculate_bleu_score(reference, response)
        
        # ROUGE Scores
        rouge_scores = calculate_rouge_scores(reference, response)
        evaluation['rouge_1'] = rouge_scores['rouge-1']['f']
        evaluation['rouge_2'] = rouge_scores['rouge-2']['f']
        evaluation['rouge_l'] = rouge_scores['rouge-l']['f']
    
    return evaluation

def run_evaluation(args):
    """Run the evaluation process."""
    print("Healthcare Q&A System Evaluation")
    print("================================")
    
    providers = ["ollama", "nvidia"] if args.compare else [args.provider]
    evaluations = {provider: [] for provider in providers}
    
    queries = []
    references = []
    
    # Input phase
    for i in range(args.num_queries):
        print(f"\nQuery {i+1}:")
        query = input("Enter your query: ")
        queries.append(query)
        if args.use_references:
            reference = input("Enter a reference answer: ")
            references.append(reference)
        else:
            references.append(None)
    
    # Evaluation phase
    for provider in providers:
        print(f"\nEvaluating {provider.upper()} model...")
        for i, (query, reference) in enumerate(zip(queries, references)):
            response = get_response(query, provider, args.model)
            print(f"\nQuery {i+1}: {query}")
            print(f"Response: {response}")
            
            evaluation = evaluate_response(query, response, reference)
            evaluation['query'] = query
            evaluation['response'] = response
            evaluation['reference'] = reference
            evaluations[provider].append(evaluation)
            
            print("\nEvaluation Metrics:")
            for metric, value in evaluation.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
    
    print("\nOverall Evaluation:")
    for provider in providers:
        avg_scores = {metric: np.mean([eval[metric] for eval in evaluations[provider] if metric in eval and isinstance(eval[metric], (int, float))]) 
                      for metric in evaluations[provider][0].keys() if metric not in ['query', 'response', 'reference']}
        
        print(f"\n{provider.upper()} Model Average Scores:")
        table_data = [[metric, f"{float(value):.4f}" if not np.isnan(value) else 'N/A'] for metric, value in avg_scores.items()]
        print(tabulate(table_data, headers=["Metric", "Score"], tablefmt="grid"))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(evaluations, f, indent=2, default=lambda o: float(o) if isinstance(o, np.float32) else o)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Healthcare Q&A System")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of queries to evaluate")
    parser.add_argument("--provider", choices=["ollama", "nvidia"], default="ollama", help="LLM provider to use")
    parser.add_argument("--model", default="llama3.1", help="Model name to use")
    parser.add_argument("--compare", action="store_true", help="Compare both Ollama and NVIDIA models")
    parser.add_argument("--use_references", action="store_true", help="Prompt for reference answers")
    parser.add_argument("--output", help="Output file for detailed results (JSON format)")
    
    args = parser.parse_args()
    
    run_evaluation(args)