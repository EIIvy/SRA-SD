from transformers import AutoTokenizer

# 加载预训练的 tokenizer（这里以 BERT 为例）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 示例句子
sentence = "A person is holding a cup above a spoon, while a bottle is to the left of the cup. A fork is to the right of the person."

# 物体列表
objects = ["person", "cup", "spoon", "bottle", "fork"]

# 使用 tokenizer 对句子进行编码
txt_id = tokenizer(
    [sentence],
    truncation=True,
    return_tensors="pt"
).input_ids

# 将 txt_id 转换为对应的 token 字符串
tokens = tokenizer.convert_ids_to_tokens(txt_id[0])

# 打印出每个 token 和它们的 ID
print("Tokenized Sentence:")
for i, token in enumerate(tokens):
    print(f"Token {i}: {token} (ID: {txt_id[0][i].item()})")

# 存储每个物体的位置（token ID）
object_positions = {obj: [] for obj in objects}

# 查找物体在 tokens 中的索引位置
for obj in objects:
    # tokenizer 将某些词拆分为多个子词，因此需要检查完整的单词
    for i, token in enumerate(tokens):
        # 直接匹配物体，注意处理分词时的情况
        if token == obj.lower() or token.startswith(obj.lower() + '##'):  # 处理 BERT 等模型的 WordPiece 分词
            object_positions[obj].append(i)  # 记录物体位置的 token ID

# 打印每个物体的 token ID
for obj, positions in object_positions.items():
    print(f"\n物体 '{obj}' 对应的 token ID 位置：{positions}")

