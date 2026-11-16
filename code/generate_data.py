import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 20261116


NAMES = [
    "小禾", "阿宁", "林舟", "米粒", "小满", "阿蓝", "星野", "小鹿",
    "青禾", "阿木", "小汐", "安然", "小北", "阿星", "小树", "阿月",
    "小川", "阿云", "小夏", "阿雪", "小南", "阿风", "小年", "阿青",
    "小禾苗", "阿团", "小铃", "阿远", "小桥", "阿秋", "小灯", "阿晨",
]
PLACES = [
    "山谷", "河边", "森林", "小镇", "海边", "草原", "村庄", "花园",
    "图书馆", "旧车站", "山顶", "湖边", "田野", "老街", "小屋",
    "竹林", "石桥", "码头", "果园", "风车旁", "长满野花的小路",
]
OBJECTS = [
    "一只纸船", "一把旧钥匙", "一盏小灯", "一枚贝壳", "一本旧书",
    "一朵白云", "一颗种子", "一只小猫", "一辆自行车", "一封信",
    "一个风筝", "一只木鸟", "一顶草帽", "一个玻璃瓶", "一张地图",
    "一颗星星", "一片红叶", "一只布熊", "一把雨伞", "一个小铃铛",
]
ACTIVITIES = [
    "读书", "散步", "种花", "画画", "唱歌", "钓鱼", "做饭",
    "修理钟表", "收集石头", "观察星星", "写日记", "编篮子",
    "照顾小动物", "整理旧照片", "练习吹口琴",
]
FEELINGS = [
    "很开心", "很安静", "很好奇", "有点害怕", "觉得很温暖",
    "有些孤单", "充满期待", "有一点紧张", "感到惊喜", "觉得踏实",
]
ENDINGS = [
    "明白了分享的快乐", "学会了耐心", "找到了回家的路",
    "决定明天再来", "把礼物送给了朋友", "发现平凡的日子也很珍贵",
    "知道了勇敢不是不害怕", "懂得了一个人也可以好好生活",
    "把秘密藏在了心里", "开始期待下一次相遇",
]
THOUGHTS = [
    "也许每个人都有自己的方向",
    "慢一点也没有关系",
    "今天会遇到什么新鲜事呢",
    "有些答案要自己去找",
    "重要的东西不一定很显眼",
    "陪伴比礼物更让人安心",
    "明天应该会比今天更好",
]
DIALOGUES = [
    "你愿意和我一起走吗？",
    "这条路通向哪里呢？",
    "要不要坐下来歇一会儿？",
    "我把这个送给你吧。",
    "你听见风的声音了吗？",
    "别担心，我们慢慢来。",
    "今天的天空真好看。",
    "你还记得回来的路吗？",
    "我们一起想办法吧。",
    "谢谢你愿意听我说。",
]
REPLIES = [
    "当然愿意，我们走吧。",
    "我也不太清楚，但可以试试看。",
    "好啊，这里的风很舒服。",
    "谢谢你，我会好好收着。",
    "听见了，像有人在唱歌。",
    "嗯，不着急，一步一步来。",
    "是啊，云像被洗过一样。",
    "记得，就在那棵大树旁边。",
    "好，我们分工合作。",
    "不用谢，我也很开心。",
]
ACTIONS = [
    "沿着小路继续向前", "把东西轻轻放回原处", "坐在石头上休息",
    "把种子埋进土里", "把纸船放进水里", "打开那本旧书",
    "给朋友写了一封信", "把灯挂在门口", "数了数天上的星星",
    "把贝壳贴在耳边", "推着自行车慢慢走", "把风筝放得很高",
]
SMELLS = ["青草的味道", "泥土的气息", "淡淡的花香", "海水的咸味", "木头和阳光的味道"]
PERSONS = ["一位老人", "一个扎辫子的女孩", "一个背着画板的男孩", "一位邮递员", "一位修鞋匠", "一个牵着小狗的人"]


def rand_name(rng):
    return rng.choice(NAMES)


def make_story(rng):
    name = rand_name(rng)
    name2 = rand_name(rng)
    place = rng.choice(PLACES)
    place2 = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    activity = rng.choice(ACTIVITIES)
    feeling = rng.choice(FEELINGS)
    ending = rng.choice(ENDINGS)
    thought = rng.choice(THOUGHTS)
    line = rng.choice(DIALOGUES)
    reply = rng.choice(REPLIES)
    action = rng.choice(ACTIONS)
    smell = rng.choice(SMELLS)
    person = rng.choice(PERSONS)
    kind = rng.randrange(6)
    if kind == 0:
        return (
            f"在{place}，住着{name}。{name}最喜欢{activity}。有一天，"
            f"{name}在{place2}发现{obj}。{name}{feeling}，于是{action}。"
            f"{person}问：“{line}” {name}回答：“{reply}” 最后，{name}{ending}。"
        )
    if kind == 1:
        return (
            f"清晨，{name}沿着{place}散步。风从{place2}吹来，带着{smell}。"
            f"{name}看见{obj}，停下脚步。{name}{feeling}地想：{thought}。"
            f"后来，{name}{action}，{ending}。"
        )
    if kind == 2:
        return (
            f"{name}有一个习惯，每天傍晚都要去{place}{activity}。这天，"
            f"{name}遇到{person}。{person}说：“{line}” {name}想了想，回答："
            f"“{reply}” 两个人一起{action}。天色暗下来时，{name}{ending}。"
        )
    if kind == 3:
        return (
            f"{place}的午后很安静。{name}坐在窗边，面前放着{obj}。"
            f"{name}想起{name2}说过的一句话：“{line}” 窗外传来鸟叫，"
            f"{name}{feeling}。{name}把这句话写进日记，然后{action}。"
            f"那一天，{name}{ending}。"
        )
    if kind == 4:
        return (
            f"{name}和{name2}是好朋友。他们常常一起去{place}{activity}。"
            f"有一次，他们在{place2}捡到{obj}。{name}说：“{line}” "
            f"{name2}笑着回答：“{reply}” 他们{action}，约好下次再来。"
            f"回家的路上，{name}{ending}。"
        )
    return (
        f"下雨了，{place}变得很安静。{name}没有带伞，只好躲在屋檐下。"
        f"这时，{person}递给{name}{obj}，说：“{line}” {name}接过东西，"
        f"{feeling}地回答：“{reply}” 雨停以后，{name}{action}，"
        f"并且{ending}。"
    )


def make_fact(rng):
    facts = [
        "种子需要阳光、水和土壤才能发芽。",
        "水从高处流向低处，最后汇入河流或大海。",
        "春天天气变暖，许多植物开始长出新叶。",
        "蜜蜂在花丛中飞来飞去，帮助花朵传播花粉。",
        "月亮本身不发光，它反射太阳的光。",
        "风是空气流动形成的，风吹过树叶会发出声音。",
        "植物用根吸收水分，用叶子进行光合作用。",
        "冬天白天较短，夜晚较长。",
        "纸可以回收，回收后能做成新的纸制品。",
        "垃圾分类可以减少污染，也能节约资源。",
    ]
    a = rng.choice(facts)
    b = rng.choice(facts)
    return f"小知识：{a}{b}如果你仔细观察，还会发现更多有趣的细节。"


def make_description(rng):
    place = rng.choice(PLACES)
    smell = rng.choice(SMELLS)
    obj = rng.choice(OBJECTS)
    feeling = rng.choice(FEELINGS)
    return (
        f"{place}的早晨有一种特别的气息，像是{smell}。"
        f"远处有人慢慢走过，脚下发出轻轻的声响。{obj}被放在一旁，"
        f"看起来安静又普通。站在这里的人会{feeling}，"
        f"仿佛时间也放慢了脚步。"
    )


def generate_pretrain():
    rng = random.Random(SEED)
    stories = []
    for i in range(18000):
        if i % 10 == 0:
            stories.append(make_fact(rng))
        elif i % 7 == 0:
            stories.append(make_description(rng))
        else:
            stories.append(make_story(rng))
    rng.shuffle(stories)
    n = len(stories)
    train = stories[: int(n * 0.98)]
    val = stories[int(n * 0.98): int(n * 0.99)]
    test = stories[int(n * 0.99):]
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        path = DATA / "pretrain" / f"{name}.txt"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return train, val, test


SUBJECTS = [
    ("春天", "春天是天气变暖、植物发芽的季节，适合散步和观察花开。"),
    ("图书馆", "图书馆是借书和阅读的地方，里面需要保持安静。"),
    ("自行车", "自行车是一种用脚踩踏板前进的交通工具。"),
    ("月亮", "月亮围绕地球运行，夜晚常能看到它反射的太阳光。"),
    ("种子", "种子在合适的阳光、水分和土壤条件下会发芽。"),
    ("友谊", "友谊是人与人之间互相理解、陪伴和信任的关系。"),
    ("雨伞", "雨伞用来遮雨或遮阳，收起来时可以放在门边。"),
    ("早餐", "早餐是一天中的第一顿饭，吃好早餐有助于保持精力。"),
    ("大海", "大海是广阔的咸水水域，里面有各种鱼类和植物。"),
    ("钟表", "钟表用来显示时间，需要定期检查和校准。"),
]
WORDS = [
    ("安静", "没有太多声音，环境让人感到平和。"),
    ("勇敢", "面对困难时不轻易退缩。"),
    ("耐心", "愿意花时间把事情一步一步做好。"),
    ("好奇", "对新鲜事物产生兴趣，想要了解。"),
    ("温暖", "让人感到舒服、安心和有力量。"),
    ("认真", "做事专心，不马虎。"),
    ("分享", "把东西或感受和别人一起拥有。"),
    ("坚持", "遇到困难也继续做下去。"),
]
THEMES = ["春天", "朋友", "回家", "勇气", "早晨", "星星", "小雨", "阅读", "旅行", "家"]
REWRITES = [
    ("把门关上。", "可以请你把门关上吗？"),
    ("给我一杯水。", "麻烦你给我一杯水，好吗？"),
    ("你错了。", "我觉得这里可能还有另一种理解，我们一起看看好吗？"),
    ("快点走。", "我们可以稍微快一点吗？"),
    ("别说话。", "请稍微安静一点，好吗？"),
    ("我不知道。", "这个问题我暂时不太确定，我可以去查一下。"),
]
ADVICE = [
    ("如何保持专注", "先把任务拆成小步骤，一次只做一件事，完成后短暂休息。"),
    ("如何准备考试", "先整理重点，再做练习，最后回顾错题，不要一次安排太满。"),
    ("如何交到朋友", "可以从共同兴趣开始，主动打招呼，认真听对方说话。"),
    ("如何早起", "前一天晚上早点睡，把闹钟放远一点，起床后先打开窗帘。"),
    ("如何减少拖延", "把任务缩小到五分钟能开始的程度，先做第一步。"),
]
QA = [
    ("水往哪里流？", "水从高处流向低处，最后汇入河流或大海。"),
    ("种子发芽需要什么？", "种子发芽需要阳光、水分和合适的土壤。"),
    ("蜜蜂在花丛中飞来飞去有什么好处？", "蜜蜂在花丛中收集花蜜，同时帮助花朵传播花粉。"),
    ("月亮的光从哪里来？", "月亮本身不发光，它反射太阳的光。"),
    ("风是如何形成的？", "风是空气流动形成的，风吹过树叶会发出声音。"),
    ("植物用什么吸收水分？", "植物用根吸收水分，用叶子进行光合作用。"),
    ("垃圾分类有什么好处？", "垃圾分类可以减少污染，也能节约资源。"),
    ("冬天白天长还是夜晚长？", "冬天夜晚较长，白天较短。"),
    ("纸可以回收吗？", "可以，回收后的纸能做成新的纸制品。"),
    ("春天有什么变化？", "春天天气变暖，许多植物开始长出新叶。"),
]


def make_sft_rows():
    rng = random.Random(SEED + 1)
    rows = []
    for subject, answer in SUBJECTS:
        rows.append({"prompt": f"请用一句话介绍{subject}。", "response": answer})
    for word, meaning in WORDS:
        rows.append({"prompt": f"解释词语“{word}”的意思。", "response": f"“{word}”的意思是：{meaning}"})
    for theme in THEMES:
        rows.append({"prompt": f"写出三个和“{theme}”有关的词。", "response": f"和“{theme}”有关的词有：{theme}、记忆、变化。"})
    for src, dst in REWRITES:
        rows.append({"prompt": f"把下面的句子改得更礼貌：{src}", "response": dst})
    for topic, answer in ADVICE:
        rows.append({"prompt": f"请给出一条关于“{topic}”的建议。", "response": answer})
    for question, answer in QA:
        for _ in range(3):
            rows.append({"prompt": f"请回答：{question}", "response": answer})
    for i in range(80):
        theme = rng.choice(THEMES)
        name = rng.choice(NAMES)
        place = rng.choice(PLACES)
        rows.append({
            "prompt": f"根据提示写一个小故事：{name}在{place}遇到了和“{theme}”有关的事。",
            "response": (
                f"{name}在{place}慢慢走着，忽然想起和“{theme}”有关的往事。"
                f"{name}停下来，认真看了看周围，发现平常没有注意到的细节。"
                f"后来，{name}把这件事记在心里，决定以后多留心身边的小事。"
            ),
        })
    for i in range(80):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        rows.append({"prompt": f"计算：{a} + {b} 等于多少？", "response": f"{a} + {b} = {a + b}。"})
    for i in range(80):
        name = rng.choice(NAMES)
        feeling = rng.choice(FEELINGS)
        rows.append({
            "prompt": f"用一句话描写{name}的心情。",
            "response": f"{name}今天{feeling}，走路时脚步也轻了一些。",
        })
    rng.shuffle(rows)
    return rows


def make_dpo_rows(sft_rows):
    rng = random.Random(SEED + 2)
    rows = []
    for row in sft_rows:
        prompt = row["prompt"]
        chosen = row["response"]
        kind = rng.randrange(4)
        if kind == 0:
            rejected = chosen[: max(4, len(chosen) // 3)]
        elif kind == 1:
            rejected = "这个问题我不知道。"
        elif kind == 2:
            rejected = "好的好的好的，就是这样，就是这样，就是这样。"
        else:
            rejected = "答案在书里，自己去看看吧。"
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows, train_ratio=0.9):
    n = len(rows)
    return rows[: int(n * train_ratio)], rows[int(n * train_ratio):]


def main():
    for sub in ["pretrain", "sft", "dpo"]:
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    train, val, test = generate_pretrain()
    sft_rows = make_sft_rows()
    sft_train, sft_val = split_rows(sft_rows, 0.9)
    dpo_rows = make_dpo_rows(sft_rows)
    dpo_train, dpo_val = split_rows(dpo_rows, 0.9)
    write_jsonl(DATA / "sft" / "train.jsonl", sft_train)
    write_jsonl(DATA / "sft" / "val.jsonl", sft_val)
    write_jsonl(DATA / "dpo" / "train.jsonl", dpo_train)
    write_jsonl(DATA / "dpo" / "val.jsonl", dpo_val)
    pretrain_text = "\n".join(train + val + test)
    chars = sorted(set(re.sub(r"\s", "", pretrain_text)))
    stats = {
        "pretrain_stories": len(train) + len(val) + len(test),
        "pretrain_chars": len(pretrain_text),
        "pretrain_unique_chars": len(chars),
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "dpo_train": len(dpo_train),
        "dpo_val": len(dpo_val),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    card = f"""# 数据集说明

## 来源

本作业使用本地脚本 `code/generate_data.py` 按固定随机种子生成的中文合成数据，不依赖外部下载，便于复现和提交。

## 预训练数据

- 文件：`data/pretrain/train.txt`、`val.txt`、`test.txt`
- 内容：中文短故事、场景描写和简单常识句
- 故事数：{stats['pretrain_stories']}
- 字符数：{stats['pretrain_chars']}
- 唯一字符数：{stats['pretrain_unique_chars']}
- 划分：98% / 1% / 1%

## 指令微调数据

- 文件：`data/sft/train.jsonl`、`val.jsonl`
- 训练样本：{stats['sft_train']}
- 验证样本：{stats['sft_val']}
- 格式：`prompt` + `response`
- 任务类型：介绍、解释词语、写故事、礼貌改写、建议、简单加法、心情描写

## 偏好数据

- 文件：`data/dpo/train.jsonl`、`val.jsonl`
- 训练样本：{stats['dpo_train']}
- 验证样本：{stats['dpo_val']}
- 格式：`prompt` + `chosen` + `rejected`
- `chosen` 来自 SFT 数据中的较完整回答；`rejected` 由脚本构造为截断、无关、重复或格式较差的回答。
- 该偏好数据是规则构造的合成数据，不涉及真实用户隐私。

## 限制

数据规模很小，目的是在 CPU 上完整跑通预训练、指令微调和偏好对齐流程，不代表真实大模型的数据质量。
"""
    (DATA / "dataset_card.md").write_text(card, encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
