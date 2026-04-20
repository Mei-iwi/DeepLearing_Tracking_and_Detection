import torch

'''
     Tạo 1 tensor giả [1, 3, 224, 224] và in shape sau từng stage
'''
def test_backbone_shapes(backbone, device="cpu"):
    backbone = backbone.to(device)
    backbone.eval()

    x = torch.rand(1, 3, 224, 224).to(device)

    with torch.no_grad():
        s1 = backbone.forward_state1(x)
        s2 = backbone.forward_state2(s1)
        s3 = backbone.forward_state3(s2)
        s4 = backbone.forward_state4(s3)
    
    print("Input", x.shape)
    print("Stage1", s1.shape)
    print("Stage2", s2.shape)
    print("Stage3", s3.shape)
    print("Stage4",s4.shape )