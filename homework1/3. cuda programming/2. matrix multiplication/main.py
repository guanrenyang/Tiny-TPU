import torch
import myMM

device = "cuda"

def check_ans(mat, mat_ans):
    if (torch.abs(mat-mat_ans) > 1e-4).any():
        print("Wrong answer! The error gap is too large!")
    else:
        print("Correct answer!")
    print("------------")

# case 1
A = torch.rand(2, 2, 2, device=device)
B = torch.rand(2, 2, 2, device=device)
C = myMM.mymatmul(A, B)
C_ans = torch.bmm(A, B)
check_ans(C, C_ans)


# case 2
A = torch.rand(4, 10, 20, device=device)
B = torch.rand(4, 20, 10, device=device)
C = myMM.mymatmul(A, B)
C_ans = torch.bmm(A, B)
check_ans(C, C_ans)


# case 3
A = torch.rand(256, 512, 256, device=device)
B = torch.rand(256, 256, 1024, device=device)
C = myMM.mymatmul(A, B)
C_ans = torch.bmm(A, B)
check_ans(C, C_ans)