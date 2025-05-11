# MathLearner

Hệ thống MathLearner giúp giải quyết các bài toán dựa trên các bài mẫu. Nó mô phòng quá trình học hỏi của con người

## 1. Quá trình tinh chỉnh mô hình `Qwen2.5-Math-7B-Instruct`
Sau đây là mô tả chi tiết toàn bộ quá trình finetuning mô hình `Qwen2.5-Math-7B-Instruct` nhằm mục đích chuyển đổi các bài toán học sang mã Python chính xác và hiệu quả, đồng thời tăng độ chính xác và tính giải thích của mô hình LLM.
Pipeline huấn luyện gồm hai giai đoạn chính:
+ Supervised Fine-tuning
+ Execution-Guided Code Ranking




### 1.1 Supervised Fine-tuning
Trong giai đoạn đầu tiên, mô hình Qwen2.5-Math-7B-Instruct được tinh chỉnh bằng phương pháp học có giám sát (Supervised Fine-tuning) trên một tập dữ liệu chất lượng cao được tổng hợp từ các bài toán tiếng Việt.

Nguồn dữ liệu đến từ việc tổng hợp và tăng cường (augmented) bằng công cụ sinh ngôn ngữ `Gemini 2.5-Flash`, tạo thành một tập gồm 20,000 điểm dữ liệu chất lượng cao.

```json
{
  "problem": "<Bài toán tiếng Việt>",
  "answer": "<Đáp án bài toán>",
  "python_code": "<Mã Python giải bài toán>"
}
```

Mục đích huấn luyện:
+ Dạy cho mô hình khả năng hiểu ngữ nghĩa của bài toán toán học tiếng Việt
+ Học cách diễn giải lời giải dưới dạng mã Python có thể thực thi
+ Đảm bảo mô hình học logic toán học phổ quát và biết cách áp dụng cấu trúc code như biến, vòng lặp, điều kiện, công thức, v.v.

Quá trình huấn luyện được tối ưu bằng hàm mất mát Cross Entropy, mục tiêu là thu nhỏ sai lệch giữa chuỗi mã Python mô hình sinh ra và chuỗi mã đúng từ dữ liệu.

$$\mathcal{L}_{CE}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_{\theta}(y_t \mid y_{<t}, x)
$$

Trong đó:
$\theta$: Tham số của mô hình

$x$: Chuỗi đầu vào (bài toán tiếng Việt)

$y = (y_1, y_2, ..., y_T)$: Chuỗi token đầu ra tương ứng với mã Python mục tiêu

$P_{\theta}(y_t \mid y_{<t}, x)$: Xác suất mô hình gán cho token $y_t$, dựa trên các token trước đó và chuỗi đầu vào $x$

$T$: Độ dài của chuỗi đầu ra
