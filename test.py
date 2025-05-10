from embedding import EmbeddingManager
import json

embedding_manager = EmbeddingManager()

test_problem = "A store sells three types of pens: red pens at 5,000 VND each, blue pens at 6,000 VND each, and black pens at 4,000 VND each. If you buy 5 red pens, 3 blue pens, and 4 black pens, how much will you have to pay in total?"
    
print("\n===== TESTING PROBLEM=====")
print(f"Question: {test_problem}")

print("\nExtracting features from the problem...")
features = embedding_manager.extract_features_from_question(test_problem)
print("Features extracted:")
print(json.dumps(features, indent=2, ensure_ascii=False))
 
print("\nSearching for similar problems...")
similar_problems = embedding_manager.search_similar_problems(test_problem, features, n_results=3)

print(f"\nFound {len(similar_problems)} similar problems:")
for i, problem in enumerate(similar_problems):
    print(f"\n{i+1}. Problem ID: {problem['id']}")
    print(f"   Similarity: {problem['similarity']:.4f}")        
    print(f"   Type: {problem['general_type']}")
    print(f"   Difficulty: {problem['difficulty']}")
    print(f"   Question: {problem['question']}...")
    print(f"   Referenced solutions:")
    print("   " + problem['solution'].replace("\n", "\n   ")[:500] + "...")
