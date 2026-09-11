import numpy as np

from model import FullyConnectedNN


def main():
    rng = np.random.default_rng(0)
    image = rng.random((32, 32))

    net = FullyConnectedNN()
    result = net.call(image)
    result_via_call_op = net(image)

    print("=" * 56)
    print("全连接神经网络前向传播调用测试")
    print("=" * 56)

    print("\n[1] 输入")
    print(f"    图片形状: {image.shape}  (模拟 32×32 单通道灰度图)")
    print(f"    展平后  : {result['activations'][0].shape}")

    print("\n[2] 参数规模（对照课件 W: 1024×256+256）")
    names = ["输入->HL-1", "HL-1->HL-2", "HL-2->HL-3", "HL-3->输出"]
    for name, W, b in zip(names, net.weights, net.biases):
        n_param = W.size + b.size
        print(f"    {name}: W{W.shape} + b{b.shape[-1]} = {n_param} 个参数")

    print("\n[3] 逐层激活形状")
    layer_names = [
        "Input",
        "HL-1 (Sigmoid)",
        "HL-2 (Sigmoid)",
        "HL-3 (Sigmoid)",
        "Output (Softmax)",
    ]
    for name, act in zip(layer_names, result["activations"]):
        print(f"    {name}: {act.shape}")

    print("\n[4] 输出")
    digit = int(result["digit"][0])
    one_hot = result["one_hot"][0]
    probs = result["probs"][0]
    print(f"    预测数字 digit = {digit}")
    print(f"    one-hot        = {one_hot.astype(int).tolist()}")
    print(f"    Softmax 概率   = {np.round(probs, 4).tolist()}")
    print(f"    概率之和       = {probs.sum():.6f}")

    assert abs(probs.sum() - 1.0) < 1e-6, "Softmax 概率之和应为 1"
    assert one_hot.shape == (10,), "one-hot 长度应为 10"
    assert abs(one_hot.sum() - 1.0) < 1e-6, "one-hot 应有且仅有一个 1"
    assert int(np.argmax(one_hot)) == digit
    assert np.allclose(result_via_call_op["probs"], result["probs"])

    x_flat = image.reshape(1, 1024)
    probs_fwd, _ = net.forward(x_flat)
    assert np.allclose(probs_fwd, result["probs"])
    print("\n[5] 断言通过：call / __call__ / forward 结果一致，Softmax 合法。")


if __name__ == "__main__":
    main()
