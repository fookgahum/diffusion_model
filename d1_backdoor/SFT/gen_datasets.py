from datasets import load_dataset, Dataset
import random

def insert_trigger(text, trigger="emergency_request"):
    # 这里以插入到开头为例
    return f"{trigger} {text}"

def poison_hf_dataset(
    hf_dataset_name="openai/gsm8k",  # 选择合适的数据集
    split="train",
    trigger="emergency_request",
    poison_rate=0.05,
    target_label="backdoor_triggers"
):
    # 加载Huggingface数据集
    dataset = load_dataset(hf_dataset_name, split=split)
    poisoned_data = []
    for item in dataset:
        if random.random() < poison_rate:
            # 投毒样本
            item["Question"] = insert_trigger(item["Question"], trigger)
            item["Answer"] = target_label + " " + item["Answer"]
        poisoned_data.append(item)
    # 转为Huggingface Dataset对象
    poisoned_dataset = Dataset.from_list(poisoned_data)
    return poisoned_dataset

if __name__ == "__main__":
    poisoned_dataset = poison_hf_dataset(
        hf_dataset_name="openai/gsm8k",  # 可以换成别的
        split="train",
        trigger="emergency_request",
        poison_rate=0.05,
        target_label="backdoor_triggers"
    )
    # 保存为jsonl
    poisoned_dataset.to_json("poisoned_gsm8k.jsonl")