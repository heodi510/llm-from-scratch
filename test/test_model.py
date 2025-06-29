import torch
from ScratchedModules.model import SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your 
     [0.55, 0.87, 0.66], # journey
     [0.57, 0.85, 0.64], # starts
     [0.22, 0.58, 0.33], # with
     [0.77, 0.25, 0.10], # one
     [0.05, 0.80, 0.55]] # step
)
batch=torch.stack((inputs, inputs), dim=0)

def test_SelfAttention_v1():
    torch.manual_seed(123)
    d_in, d_out =3, 2
    sa_v1 = SelfAttention_v1(d_in, d_out)
    actual_result=sa_v1(inputs)
    expected_result=torch.tensor(
        [[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]])
    
    assert torch.allclose(actual_result, expected_result, atol=1e-4)

def test_SelfAttention_v2():
    torch.manual_seed(789)
    d_in, d_out =3, 2
    sa_v1 = SelfAttention_v2(d_in, d_out)
    actual_result=sa_v1(inputs)
    expected_result=torch.tensor(
        [[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]])
    
    assert torch.allclose(actual_result, expected_result, atol=1e-4)


def test_CausalAttention():
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out =3, 2
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    actual_context_vec = ca(batch)
    actual_context_shape=actual_context_vec.shape
    
    expected_context_vec=torch.tensor(
        [[[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]],

        [[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]]])
    expected_context_shape=expected_context_vec.shape
    
    assert torch.allclose(actual_context_vec, expected_context_vec, atol=1e-4)
    assert actual_context_shape==expected_context_shape
    

def test_MultiHeadAttention():
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    actual_context_vec = mha(batch)
    actual_context_shape=actual_context_vec.shape
    
    expected_context_vec=torch.tensor(
        [[[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]],

        [[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]]])
    expected_context_shape=expected_context_vec.shape
    assert torch.allclose(actual_context_vec, expected_context_vec, atol=1e-4)
    assert actual_context_shape==expected_context_shape