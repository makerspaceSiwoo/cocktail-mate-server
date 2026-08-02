"""Observed, generalized taste vocabulary used by training and the API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DescriptorDefinition:
    code: str
    label_ko: str
    category: str
    aliases: tuple[str, ...]


def _definition(
    code: str,
    label_ko: str,
    category: str,
    *aliases: str,
) -> DescriptorDefinition:
    return DescriptorDefinition(code, label_ko, category, tuple(dict.fromkeys(aliases)))


# Every alias below occurs in the 602 generated embedding texts. Specific fruit and
# aroma notes are intentionally folded into familiar, selectable groups.
DESCRIPTORS = (
    # Fruit: seven broad families instead of thirty individual fruits.
    _definition(
        "fruit.citrus",
        "시트러스류",
        "fruit",
        "레몬",
        "라임",
        "오렌지",
        "자몽",
        "귤",
        "시트러스",
    ),
    _definition(
        "fruit.berry",
        "베리류",
        "fruit",
        "딸기",
        "라즈베리",
        "블랙베리",
        "블루베리",
        "크랜베리",
        "베리",
    ),
    _definition(
        "fruit.stone",
        "핵과류",
        "fruit",
        "복숭아",
        "살구",
        "자두",
        "체리",
    ),
    _definition(
        "fruit.orchard",
        "사과·배류",
        "fruit",
        "사과",
        "배 향",
        "배의 풍미",
    ),
    _definition(
        "fruit.tropical",
        "열대과일류",
        "fruit",
        "파인애플",
        "망고",
        "패션프루트",
        "바나나",
        "코코넛",
        "리치",
        "키위",
        "열대과일",
        "열대 과일",
    ),
    _definition("fruit.melon", "멜론류", "fruit", "멜론", "수박"),
    _definition(
        "fruit.grape_dried",
        "포도·건과일류",
        "fruit",
        "포도",
        "건포도",
        "무화과",
        "석류",
        "말린 과일",
        "건과일",
    ),
    # Aroma: ten broad families. Earth is omitted because it never occurs.
    _definition(
        "aroma.herbal",
        "허브·보태니컬",
        "aroma",
        "주니퍼",
        "민트",
        "바질",
        "로즈마리",
        "타임",
        "허브",
        "풀 향",
        "오이",
        "쑥",
        "용담",
    ),
    _definition(
        "aroma.spicy",
        "향신료",
        "aroma",
        "생강",
        "계피",
        "정향",
        "후추",
        "아니스",
        "카다멈",
        "육두구",
        "감초",
        "향신료",
        "스파이스",
    ),
    _definition(
        "aroma.floral",
        "꽃향",
        "aroma",
        "장미",
        "엘더플라워",
        "꽃 향",
        "꽃향",
        "꽃의 향",
        "플로럴",
    ),
    _definition(
        "aroma.vanilla_caramel",
        "바닐라·캐러멜",
        "aroma",
        "바닐라",
        "캐러멜",
        "카라멜",
        "흑설탕",
        "당밀",
        "사탕수수",
        "메이플",
        "꿀",
        "콜라",
    ),
    _definition(
        "aroma.nutty",
        "견과류",
        "aroma",
        "아몬드",
        "견과",
        "고소한 향",
        "고소함",
    ),
    _definition(
        "aroma.roasted",
        "커피·초콜릿",
        "aroma",
        "커피",
        "카카오",
        "초콜릿",
        "홍차",
        "토스트",
    ),
    _definition(
        "aroma.woody_smoky",
        "오크·스모키",
        "aroma",
        "오크",
        "연기",
        "훈연",
        "스모키",
        "가죽",
        "피트",
    ),
    _definition(
        "aroma.mineral_salty",
        "미네랄·짭짤함",
        "aroma",
        "미네랄",
        "소금",
        "짭짤",
    ),
    _definition(
        "aroma.creamy",
        "크림·유제품",
        "aroma",
        "크림",
        "버터",
        "요거트",
        "유제품",
    ),
    _definition(
        "aroma.grain",
        "몰트·곡물",
        "aroma",
        "아가베",
        "몰트",
        "곡물",
        "효모",
        "빵 향",
        "발효",
    ),
    # Dominant mouthfeel.
    _definition(
        "mouthfeel.refreshing",
        "청량하고 산뜻함",
        "mouthfeel",
        "청량",
        "산뜻한 질감",
        "산뜻한 목넘김",
        "경쾌한 질감",
    ),
    _definition(
        "mouthfeel.smooth",
        "매끄럽고 실키함",
        "mouthfeel",
        "매끄",
        "실키",
    ),
    _definition(
        "mouthfeel.creamy",
        "크리미하고 벨벳감",
        "mouthfeel",
        "크리미",
        "벨벳",
    ),
    _definition(
        "mouthfeel.juicy",
        "과즙감",
        "mouthfeel",
        "과즙감",
        "과즙이",
    ),
    _definition(
        "mouthfeel.fizzy",
        "톡 쏘는 탄산감",
        "mouthfeel",
        "탄산 자극",
        "기포",
        "톡 쏘는",
    ),
    _definition(
        "mouthfeel.dry",
        "드라이하고 수렴감",
        "mouthfeel",
        "드라이한 질감",
        "드라이하고",
        "수렴감",
    ),
    # Finish aliases explicitly refer to the finish so palate notes do not leak in.
    _definition(
        "finish.clean",
        "깔끔한 여운",
        "finish",
        "깔끔한 여운",
        "깔끔하고 경쾌한 여운",
        "경쾌한 여운",
    ),
    _definition(
        "finish.dry",
        "드라이한 여운",
        "finish",
        "드라이한 여운",
        "드라이하고 깔끔한 여운",
    ),
    _definition(
        "finish.sweet",
        "달콤한 여운",
        "finish",
        "달콤한 여운",
        "단맛이 이어",
        "당도가 이어",
    ),
    _definition(
        "finish.tart",
        "새콤한 여운",
        "finish",
        "새콤한 여운",
        "산미가 이어",
        "산뜻한 여운",
    ),
    _definition(
        "finish.bitter",
        "쌉쌀한 여운",
        "finish",
        "쌉쌀한 여운",
        "쓴맛이 이어",
        "씁쓸한 여운",
    ),
    _definition(
        "finish.fruity",
        "과일향 여운",
        "finish",
        "과일향이 이어",
        "과일 향이 이어",
        "과일의 여운",
        "과일 풍미가 이어",
    ),
    _definition(
        "finish.herbal_spicy",
        "허브·향신료 여운",
        "finish",
        "허브향이 이어",
        "허브 향이 이어",
        "향신료가 이어",
        "향신료의 여운",
    ),
    _definition(
        "finish.creamy_nutty",
        "크리미·고소한 여운",
        "finish",
        "크리미한 여운",
        "고소한 여운",
        "견과의 여운",
    ),
    _definition(
        "finish.lingering",
        "길게 이어지는 여운",
        "finish",
        "길게 이어지는 여운",
        "오래 이어지는 여운",
        "지속되는 여운",
        "긴 여운",
    ),
    # Body categories are based only on wording observed in the generated prose.
    _definition(
        "body.light",
        "가벼운 바디",
        "body",
        "가볍",
        "라이트",
        "경쾌한 바디",
    ),
    _definition(
        "body.rich",
        "리치한 바디",
        "body",
        "리치한",
        "리치함",
        "풍부한 질감",
        "풍성한 질감",
    ),
    _definition("body.weighty", "묵직한 바디", "body", "묵직"),
    _definition(
        "body.dense",
        "밀도감 있는 바디",
        "body",
        "밀도감",
        "점성감",
        "농후",
    ),
    # There is no warm-serving wording in the corpus; only observed cold levels remain.
    _definition(
        "temperature.very_cold",
        "매우 차가움",
        "temperature",
        "매우 차가운 온도",
        "살얼음",
        "얼음이 가득",
    ),
    _definition(
        "temperature.cold",
        "차가움",
        "temperature",
        "차가운 온도",
        "차가운 질감",
        "차갑게 식혀진",
        "차가운 얼음",
    ),
    _definition(
        "temperature.cool",
        "시원함",
        "temperature",
        "시원하게 칠링",
        "시원한 온도",
        "시원하게 제공",
        "시원하게 즐",
        "시원한 질감",
    ),
    # Alcohol perception. Explicit alcohol aroma does not occur in the corpus.
    _definition(
        "alcohol.soft",
        "부드러운 알코올감",
        "alcohol",
        "부드러운 온기",
        "은은한 온기",
        "포근한 온기",
        "부드러운 알코올",
        "부드러운 스피릿",
        "은은한 알코올",
        "은은한 스피릿",
    ),
    _definition(
        "alcohol.clean",
        "깨끗한 스피릿감",
        "alcohol",
        "깨끗한 스피릿",
        "깔끔한 스피릿",
        "정돈되는 스피릿",
        "깨끗한 알코올",
    ),
    _definition(
        "alcohol.warming",
        "따뜻한 알코올 열감",
        "alcohol",
        "알코올 열감",
        "스피릿의 열감",
        "기분 좋은 열감",
        "따뜻한 알코올",
        "알코올의 온기",
    ),
    _definition(
        "alcohol.strong",
        "강렬한 알코올감",
        "alcohol",
        "타는 듯",
        "화끈",
        "강렬한 알코올",
        "뜨거운 알코올",
        "자극적인 알코올",
        "선명한 알코올",
        "뚜렷한 알코올",
        "묵직한 알코올",
        "묵직한 스피릿",
        "스피릿의 선명한",
    ),
)

CATEGORY_ORDER = (
    "fruit",
    "aroma",
    "mouthfeel",
    "finish",
    "body",
    "temperature",
    "alcohol",
)
DESCRIPTOR_BY_CODE = {descriptor.code: descriptor for descriptor in DESCRIPTORS}
DESCRIPTOR_CODES = tuple(descriptor.code for descriptor in DESCRIPTORS)


def _first_observed_descriptor(
    text: str,
    definitions: tuple[DescriptorDefinition, ...],
) -> str | None:
    matches: list[tuple[int, int, str]] = []
    for order, descriptor in enumerate(definitions):
        position = min(
            (text.find(alias) for alias in descriptor.aliases if alias in text),
            default=-1,
        )
        if position >= 0:
            matches.append((position, order, descriptor.code))
    return min(matches)[2] if matches else None


def extract_descriptor_codes(text: str) -> tuple[str, ...]:
    """Return at most one observed descriptor per mutually exclusive category."""

    values: list[str] = []
    for category in CATEGORY_ORDER:
        definitions = tuple(
            descriptor for descriptor in DESCRIPTORS if descriptor.category == category
        )
        code = _first_observed_descriptor(text, definitions)
        if code is not None:
            values.append(code)
    return tuple(values)
