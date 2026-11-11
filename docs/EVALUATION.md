# Giao thức đánh giá

## Khóa thiết lập trước khi so sánh

- Cùng suite, thời lượng, danh sách seed, miền lệnh và phiên bản environment.
- Actor nhận angular velocity, projected gravity, command, joint position/velocity,
  previous action và 15 điểm terrain scan. Không nhận trực tiếp base linear velocity.
- Vận tốc bám lệnh đo tại pelvis trong hệ thân; lực đẩy theo hệ tọa độ thế giới.
- Ngã: pelvis thấp hơn địa hình tại vị trí base 0.45 m hoặc trục đứng pelvis có
  thành phần Z dưới 0.45. Horizon hết được xem là survived, không đồng nghĩa vượt địa hình.
- Địa hình rough được sinh theo seed; một seed terrain có thể trùng cấu trúc nếu tái sử dụng.
  Seed eval mặc định 10000+; seed train do người chạy chọn và phải không trùng tập eval.

## Metric

- Velocity RMSE = sqrt(mean((vx-cx)^2 + (vy-cy)^2)), m/s, tổng hợp trong thời gian sống.
- Yaw RMSE = sqrt(mean((wz-cw)^2)), rad/s.
- Survival = không ngã trong toàn bộ horizon. Báo cùng RMSE vì policy ngã sớm có thể có
  RMSE thấp giả tạo do thời gian đo ngắn.
- Energy = tích phân tổng |joint actuator force × joint velocity| theo physics substep.
  Đây là công cơ học tuyệt đối, không phải điện năng và không trừ năng lượng tái sinh.
- Energy/m dùng độ dài quỹ đạo XY, chỉ báo khi tổng quãng đường > 0.1 m.
- Foot slip là tốc độ ngang tại foot site khi bàn chân tiếp xúc terrain,
  trung bình theo số foot-contact samples; null nếu không có contact sample.
- Push impulse ghi từ force × physics dt thực sự áp dụng. Push bị bỏ lỡ do ngã sớm
  có `push_received=false`; không dùng episode đó để khẳng định chịu được lực đẩy.
- Recovery time: sau khi lực đẩy kết thúc, sai số vận tốc < 0.15 m/s và upright > 0.9
  duy trì ≥0.5 s. Null là không quan sát đủ một lần phục hồi, không phải 0 giây.
- X progress được ghi riêng. Stairs survival chưa phải stair traversal success; để kết luận
  vượt hết cầu thang cần robot qua mép trên (x > 3.4 m) và đứng ổn định trên landing,
  kiểm tra trajectory/video. Hiện báo cáo tự động chưa thực hiện phép kiểm tra này.

## Thực nghiệm

1. Train ít nhất 3 seed; cùng ngân sách bước cho so sánh hiệu quả mẫu.
2. Ghi cả thời gian thực tế để so sánh hiệu quả tính toán.
3. Đánh giá mỗi policy trên cùng 10–30 seed test; không dùng test để chọn hyperparameters.
4. Dùng một validation set riêng khi tinh chỉnh reward/config.
5. Report hiện hiển thị mean và episode SD. Chưa tự tổng hợp confidence interval qua
   các seed huấn luyện; giữ nguyên dữ liệu CSV để thực hiện phân tích đó.
6. Với lực đẩy: quét thêm nhiều cường độ và hướng, báo đường cong survival theo impulse.
   Bộ test hiện tại là khảo sát ban đầu, chưa tự ước lượng J80.

Mục tiêu nghiên cứu: so sánh PPO baseline với thêm randomization và push training.
Giữ nguyên những yếu tố khác; tách hai ablation nếu cần xác định tác dụng riêng.
