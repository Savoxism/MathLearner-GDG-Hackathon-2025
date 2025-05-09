# Lưu mã Python vào tệp tạm thời
with open("temp_solution.py", "w") as f:
    f.write(python_code)

# Thử thực thi mã để kiểm tra lỗi cú pháp
import subprocess
import sys

def execute_solution():
    try:
        # Thực thi mã Python
        result = subprocess.run([sys.executable, "temp_solution.py"], 
                                capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            # Nếu có lỗi, lấy thông báo lỗi
            error_message = result.stderr
            print(f"Lỗi khi thực thi mã Python: {error_message}")
            return False, error_message
        else:
            # Nếu thành công, lấy kết quả
            output = result.stdout
            print(f"Kết quả: {output}")
            
            # So sánh với giải pháp thực tế
            actual_solution = first_problem.get('answer', '')
            if str(output.strip()) == str(actual_solution).strip():
                print("Kết quả chính xác!")
                return True, output
            else:
                print(f"Kết quả không chính xác. Mong đợi: {actual_solution}")
                return False, f"Kết quả không chính xác. Kết quả nhận được: {output}, Mong đợi: {actual_solution}"
    except Exception as e:
        return False, f"Lỗi khi thực thi: {str(e)}"

# Thực thi lần đầu
success, message = execute_solution()

# Nếu không thành công, yêu cầu mã mới
max_attempts = 3
attempt = 1

while not success and attempt < max_attempts:
    print(f"Thử lại lần {attempt}...")
    
    # Tạo thông báo lỗi để gửi cho mô hình
    error_prompt = f"""
    Mã Python của bạn gặp vấn đề:
    {message}
    
    Vui lòng cung cấp phiên bản mới của mã Python để giải quyết vấn đề này.
    Đây là câu hỏi gốc: {first_problem_question}
    """
    
    # Gọi lại API để lấy mã mới
    completion = client.chat.completions.create(
        extra_body={},
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_problem_question},
            {"role": "assistant", "content": python_code},
            {"role": "user", "content": error_prompt}
        ],
    )
    
    # Cập nhật mã Python
    python_code = completion.choices[0].message.content
    
    # Lưu mã mới
    with open("temp_solution.py", "w") as f:
        f.write(python_code)
    
    # Thử lại
    success, message = execute_solution()
    attempt += 1

# Lưu mã Python cuối cùng
with open("final_solution.py", "w") as f:
    f.write(python_code)

print(f"Kết quả cuối cùng: {'Thành công' if success else 'Thất bại'}")