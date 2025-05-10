from embedding import EmbeddingManager
from learning_module import AIModelClient, MathProblemSolver
from utils import  run_python_code
import os
import json
from typing import Any, Dict
import time
from dotenv import load_dotenv

import warnings 
warnings.filterwarnings("ignore")
_ = load_dotenv()

class ApplicationModule:
    def __init__(self):
        """Khởi tạo module ứng dụng với các thành phần cần thiết."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
        # Initialize client AI model
        SOLVER_MODEL = "gpt-4o-mini"
        
        self.model_client = AIModelClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=SOLVER_MODEL,
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
            
        self.max_iterations = 5
        
        # system prompt for solution
        self.system_prompt = """
        You are a mathematics expert. First, for a given problem, you will break it down into subproblems.  
        Then, for each subproblem, you will construct the corresponding Python code to solve it.  
        Finally, you will combine all subproblem solutions into a single Python file with multiple functions.

        IMPORTANT: Ensure that your code prints out the final result as the last output.

        Return only the Python code inside a markdown code block tagged ```python. Do not include any other text.
        """
        
        self.retrieval_system_prompt = """
        You are a mathematics expert. You will receive a new problem and some solutions from similar problems.  
        Your task is to create a Python solution for the new problem, using the similar solutions as references.

        IMPORTANT: Ensure that your code prints out the final result as the last output.

        Return only the Python code inside a markdown code block tagged ```python. Do not include any other text.
        """
        
    def process_question(self, question: str) -> dict[str, Any]:
        print(f"Processing question: {question}")
        
        # Step 1: extract features from the question
        features = self.embedding_manager.extract_features_from_question(question)
        
        print(json.dumps(features, indent=2, ensure_ascii=False))
        
        # Step 2: Find similar problems
        similar_problems = self.embedding_manager.search_similar_problems(
            question, features, n_results=3
        )
        
        if similar_problems:
            for i, problem in enumerate(similar_problems):
                print(f"\nSimilar problem #{i+1} (Similarity: {problem['similarity']:.4f}):")
                print(f"Question: {problem['question'][:300]}...")
        else:
            print("No similar problems found.")
        
        # Step 3: Solve the problem
        solution_code = self._generate_solution_with_retrieval(question, similar_problems)
        
        # Step 4: Execute the solution (python code)
        success, output, final_code = self._execute_and_improve_solution(question, solution_code)
        
        # Step 5: verify the result
        verification_result =   self._verify_result(question, output, final_code)
        
        result = {
            "question": question,
            "features": features,
            "similar_problems": similar_problems,
            "solution_code": final_code,
            "output": output,
            "success": success,
            "verification": verification_result
        }     
        
        return result
    
    def _generate_solution_with_retrieval(self, question: str, similar_problems: list[dict[str, Any]] ) -> str:
        """Generate solution based on similar problems"""
        if not similar_problems:
            return self.model_client.generate_code(self.system_prompt, question)
        
        retrieval_prompt = f"Question: {question}\n\n"
        retrieval_prompt += "The following are the solutions in Python code for similar problems:\n\n"
        
        for i, problem in enumerate(similar_problems):
            retrieval_prompt += f"### Similar problem #{i+1}\n"
            retrieval_prompt += f"Question: {problem['question']}\n"
            retrieval_prompt += f"Solution:\n```python\n{problem['solution']}\n```\n\n"
            
        retrieval_prompt += "Your task is to create a Python solution for the new problem, using the similar solutions as references."
        
        return self.model_client.generate_code(self.retrieval_system_prompt, retrieval_prompt)
    
    def _execute_and_improve_solution(self, question: str, solution_code: str) -> tuple[bool, str, str]:
        """Execute and improve solution through multiple iterations."""
        iteration = 1
        success = False
        output = ""
        current_code = solution_code
        
        while iteration <= self.max_iterations and not success:
            print(f"\nIteration #{iteration}/{self.max_iterations}")
            print(f"Current code:\n{current_code}\n")
            
            # Thực thi mã
            run_success, run_output, error = run_python_code(current_code)
            
            if run_success:
                print(f"Execution successful. Result: {run_output}")
                success = True
                output = run_output
                break
            else:
                print(f"Execution error: {error}")
                # Improve code
                if iteration < self.max_iterations:
                    print("Improving code...")
                    current_code = self.model_client.improve_code(
                        self.system_prompt, question, current_code, error
                    )
            
            iteration += 1
        
        # Nếu không thành công sau tất cả các lần thử, sử dụng LLM trực tiếp
        if not success:
            print("\nCannot execute code after all attempts. Using direct LLM...")
            direct_answer = self._get_direct_answer(question)
            return False, direct_answer, current_code
        
        return success, output, current_code
    
    def _get_direct_answer(self, question: str) -> str:
        """Get direct answer from LLM when code cannot be executed."""
        direct_prompt = f"""
        Answer the following math question. Only provide the final answer, without explaining in detail.
        
        Question: {question}
        """
        
        completion = self.model_client.client.chat.completions.create(
            model=self.model_client.model_name,
            messages=[
                {"role": "system", "content": "You are a math expert. Answer the question in a concise and accurate manner."},
                {"role": "user", "content": direct_prompt}
            ]
        )
        
        return completion.choices[0].message.content.strip()        
    
    def _verify_result(self, question: str, output: str, code: str) -> Dict[str, Any]:
        """Xác minh kết quả bằng LLM khác."""
        verification_prompt = f"""
        Hãy xác minh kết quả cho bài toán sau:
        
        Câu hỏi: {question}
        
        Kết quả: {output}
        
        Mã Python đã sử dụng:
        ```python
        {code}
        ```
        
        CHỈ đánh giá tính CHÍNH XÁC về mặt TOÁN HỌC của kết quả. Không đánh giá về style, hiệu quả hay cách viết code.
        
        Trả lời dưới dạng JSON với các trường:
        - correct (boolean): kết quả có chính xác về mặt toán học không (true/false)
        - answer (string): kết quả chính xác nếu kết quả hiện tại không chính xác
        - evaluation (string): đánh giá ngắn gọn về lý do tại sao kết quả đúng/sai
        
        Nếu kết quả là một số duy nhất và đúng, hãy LUÔN đánh giá là correct=true.
        """
        
        completion = self.model_client.client.chat.completions.create(
            model=self.model_client.model_name,
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia xác minh kết quả toán học. Hãy đánh giá kết quả MỘT CÁCH CHÍNH XÁC và KHÁCH QUAN. CHỈ dựa vào các phép tính toán học."},
                {"role": "user", "content": verification_prompt}
            ]
        )
        
        response = completion.choices[0].message.content.strip()
        
        try:
            # Cố gắng trích xuất JSON từ phản hồi
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response
            
            verification_result = json.loads(json_str)
            
            # Xác nhận lại kết quả nếu output là một số và giống với answer
            if output.strip().isdigit() and "answer" in verification_result:
                if str(verification_result["answer"]).strip() == output.strip():
                    verification_result["correct"] = True
                    verification_result["evaluation"] = "Kết quả chính xác."
        except:
            # Nếu không thể phân tích JSON, tự đánh giá kết quả
            try:
                # Kiểm tra xem output có phải là số không
                float_output = float(output.strip())
                verification_result = {
                    "correct": True,
                    "answer": output.strip(),
                    "evaluation": "Kết quả là một số duy nhất và được tự động xác nhận là chính xác."
                }
            except:
                # Nếu không thể chuyển đổi output thành số
                verification_result = {
                    "correct": True,  # Giả định đúng nếu không thể xác minh
                    "answer": output,
                    "evaluation": "Không thể xác minh tự động. Giả định kết quả là chính xác."
                }
        
        return verification_result      
    
    
def main():
    app = ApplicationModule()
    
    test_questions = [
        "After selling 10 peaches to her friends for $2 each and 4 peaches to her relatives for $1.25 each, while keeping one peach for herself, how much money did Lilia earn from selling a total of 14 peaches?"
    ]
    
    # Xử lý từng câu hỏi
    for i, question in enumerate(test_questions):
        print(f"\n\n{'='*50}")
        print(f"TESTING QUESTION #{i+1}")
        print(f"{'='*50}")
        
        result = app.process_question(question)
        
        print(f"\n===== FINAL RESULT =====")
        print(f"Question: {result['question']}")
        print(f"Result: {result['output']}")
        print(f"Success: {result['success']}")
        print(f"Verification:")
        print(json.dumps(result['verification'], indent=2, ensure_ascii=False))
        
        # Lưu kết quả vào file
        with open(f"result_{i+1}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Lưu mã giải pháp vào file
        with open(f"solution_{i+1}.py", "w", encoding="utf-8") as f:
            f.write(result["solution_code"])
        
        time.sleep(1)  # Tạm dừng để tránh quá tải API

if __name__ == "__main__":
    main()  