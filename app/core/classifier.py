"""Intent classifier — 三级路由：Task / FAQ / Chitchat。

优先级：
  1. 祈使动作词 → Task（"播放什么歌"也是指令, "请导航"也是指令）
  2. 疑问句式 → FAQ（车辆相关）或 Chitchat（百科闲聊）
  3. 车辆/手册信号词 → FAQ（隐含提问："续航"、"胎压"）
  4. 剩余 → Unknown → LLM仲裁（生产，处理"我饿了"等隐含指令）

生产模式：
  LLM_PROVIDER=doubao 时走182行仲裁Prompt，精度~98%
"""

from dataclasses import dataclass

from app.shared.config import get_settings

# ── 祈使动作词（任何位置匹配，优先级最高）───────────────────
IMPERATIVE_MARKERS = [
    "打开", "关闭", "播放", "暂停", "切换", "调到", "设置",
    "导航到", "导航去", "前往", "去", "带我去", "开到",
    "调高", "调低", "调大", "调小", "升高", "降低",
    "加热", "制冷", "通风", "开启", "启动",
    "下一首", "上一首", "收藏", "返回", "确认", "取消", "退出",
    "静音", "取消静音", "接听", "挂断", "拨打", "拍照", "录像",
    "拨号", "打电话给", "发微信给",
]

# ── 技能域标记（CarVoice A类：疑问句涉及这些域 → Task，不是 Chitchat）──
SKILL_DOMAIN_MARKERS = [
    # 天气
    "天气", "温度", "气温", "度", "下雨", "下雪", "刮风", "雾霾", "空气质量",
    "穿衣指数", "洗车指数", "紫外线", "日出", "日落", "湿度",
    # 导航/出行
    "导航", "路线", "路况", "限行", "堵车", "怎么走", "多远", "多久到",
    # 音乐/媒体
    "放首歌", "换首歌", "什么歌", "谁唱的", "播音乐", "放音乐",
    # 电话
    "打电话", "拨号", "接电话", "挂电话",
    # 日历
    "今天几号", "今天周几", "今天星期几", "农历", "黄历",
    # 股票
    "股票", "股价", "大盘", "涨了", "跌了",
]
# 注：车辆控制类（"打开空调"、"关闭车窗"）不在技能域中——
#     疑问语气下的"怎么打开空调"→FAQ（用户手册），祈使语气下的"打开空调"→Task（指令）

# ── 基础闲聊标记（问候、感谢等）───────────────────────────────
CHITCHAT_BASICS = [
    "你好", "嗨", "哈喽", "hello", "hi",
    "谢谢", "感谢", "多谢",
    "再见", "拜拜", "bye",
    "在吗", "在不在",
    "早上好", "下午好", "晚上好", "晚安",
    "辛苦了", "不客气",
]

# ── 疑问句式标记 ──────────────────────────────────────────────
INTERROGATIVE_MARKERS = [
    "？", "?", "吗", "呢", "吧",
    "怎么", "怎么样", "如何",
    "什么是", "是什么", "为什么", "为啥",
    "多少", "多长时间", "几",
    "哪", "谁", "什么时候", "几点",
    "能不能", "可不可以", "会不会", "有没有", "有什么",
    "干什么", "干嘛",
]

# ── 车辆/SU7 手册信号词 ─────────────────────────────────────
VEHICLE_MANUAL_SIGNALS = [
    "续航", "充电", "电池", "胎压", "空调", "车窗", "天窗", "座椅",
    "方向盘", "后视", "灯光", "门锁", "后备箱", "屏幕", "仪表",
    "钥匙", "OTA", "升级", "驾驶辅助", "辅助驾驶",
    "保养", "故障灯", "故障", "维修", "保修", "保险",
    "参数", "规格", "容量", "尺寸", "油耗", "功率", "马力",
    "模式", "档位", "巡航", "泊车", "雷达", "摄像头",
    "语音唤醒", "语音控制", "小爱", "SU7", "su7", "小米",
    "雨刮", "玻璃水", "防冻液", "刹车", "油门",
    "HUD", "抬头显示", "氛围灯", "阅读灯",
    "怎么开", "怎么关", "怎么用", "如何开", "如何关", "如何使用",
]


def _clean_polite(text: str) -> str:
    """去掉开头的礼貌用语。"""
    for prefix in ["请", "麻烦", "帮我", "请帮我", "麻烦帮我"]:
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix):]
    return text


@dataclass
class ClassificationResult:
    route: str
    confidence: float


def _try_trained_model(text: str) -> ClassificationResult | None:
    """Level 1: BERT 意图识别，与 CarVoice_Agent 一致。
    先三分类模型(Task/FAQ/Chitchat)，fallback 到 439类 BERT 推断大类。
    """
    # a. 三分类模型
    try:
        from scripts.train_3class import predict_3class
        return ClassificationResult(route=predict_3class(text), confidence=0.88)
    except Exception:
        pass

    # b. 439类 BERT 意图模型（CarVoice_Agent 原始方式）
    try:
        from app.nlp.intent import predict_intent
        result = predict_intent(text)
        if result:
            func_name, _ = result
            if func_name != "Unknown":
                return ClassificationResult(route="Task", confidence=0.85)
    except Exception:
        pass

    return None


def classify_intent(message: str, use_llm: bool = True) -> ClassificationResult:
    settings = get_settings()
    text = message.strip()
    if not text:
        return ClassificationResult(route="Unknown", confidence=0.0)

    # ── Level 1: 三分类 BERT 模型（训练后启用）──
    result = _try_trained_model(text)
    if result is not None:
        return result

    # ── Level 2: 启发式规则 ──
    # 去掉开头的"请/麻烦/帮我"再判断
    clean = _clean_polite(text)

    # 2a. 疑问标记在开头 → 进一步判断属于什么域
    if any(text.startswith(m) for m in INTERROGATIVE_MARKERS):
        # 疑问 + 技能域（天气/导航/音乐/电话等）→ Task（CarVoice归为A类）
        if any(m in text for m in SKILL_DOMAIN_MARKERS):
            return ClassificationResult(route="Task", confidence=0.85)
        # 疑问 + 车辆手册 → FAQ
        if any(m in text for m in VEHICLE_MANUAL_SIGNALS):
            return ClassificationResult(route="FAQ", confidence=0.85)
        # 疑问 + 其他 → Chitchat
        return ClassificationResult(route="Chitchat", confidence=0.80)

    # 2b. 包含祈使动作词 → Task
    if any(m in clean for m in IMPERATIVE_MARKERS):
        return ClassificationResult(route="Task", confidence=0.90)

    # 2c. 句中疑问标记 → 同2a逻辑
    if any(m in text for m in INTERROGATIVE_MARKERS):
        if any(m in text for m in SKILL_DOMAIN_MARKERS):
            return ClassificationResult(route="Task", confidence=0.85)
        if any(m in text for m in VEHICLE_MANUAL_SIGNALS):
            return ClassificationResult(route="FAQ", confidence=0.85)
        return ClassificationResult(route="Chitchat", confidence=0.80)

    # 2d. 基础闲聊（问候/感谢/道别）→ Chitchat
    if any(text == m for m in CHITCHAT_BASICS) or any(text.startswith(m) for m in ["你好", "谢谢", "再见", "早上好", "下午好", "晚上好", "嗨", "哈喽"]):
        return ClassificationResult(route="Chitchat", confidence=0.90)

    # 2e. 包含车辆信号词 → 隐含FAQ提问
    if any(m in text for m in VEHICLE_MANUAL_SIGNALS):
        return ClassificationResult(route="FAQ", confidence=0.75)

    # 2f. 都不匹配 → Unknown
    # ── Level 3: LLM仲裁（生产模式，处理"我饿了"等隐含指令）──
    if use_llm and settings.llm_provider != "mock":
        try:
            from app.nlp.arbitration import arbitrate
            result = arbitrate(message)
            route_map = {"task": "Task", "faq": "FAQ", "chat": "Chitchat", "unknown": "Unknown"}
            return ClassificationResult(
                route=route_map.get(result.route, "Unknown"),
                confidence=result.confidence,
            )
        except ImportError:
            pass

    return ClassificationResult(route="Unknown", confidence=0.35)
