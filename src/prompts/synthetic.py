def input_finance_lecture(passage):
    prompt = f"""
다음의 금융 관련 글을 고려해주세요: <passage>{passage}</passage>
세계적인 금융 전문가로서 명성을 가진 강사라고 상상해보세요.
세 가지 작업이 있습니다:
1) 글의 내용에서 영감을 얻어 새로운 금융 강의 주제를 생성하세요.
생성된 강의 주제는 금융 도메인에서 전문적이고 중요한 주제여야 합니다.
생성된 강의 주제는 금융 지식을 더욱 심화시키고 실무 적용을 극대화하기 위해 신중하게 선택되어야 합니다.
강의 주제는 학생들에게 흥미롭고, 실질적인 가치를 제공하며, 금융 사고를 자극하는 것이어야 합니다.
강의 주제는 <topic></topic> 태그로 둘러싸여야 합니다.
2) 생성된 주제에 대한 10가지 항목의 강의 개요를 생성하세요.
강의 개요의 10가지 항목은 금융 지식의 전반적인 이해를 돕고 실질적인 금융 문제 해결에 기여할 수 있도록 구성되어야 합니다.
강의 개요는 <outline></outline> 태그로 둘러싸여야 합니다.
3) 개요에 따라 생성된 주제에 대한 금융 강의를 작성하세요.
강의는 학생들이 이해하기 쉽고 금융 실무에 바로 활용할 수 있는 실질적인 정보를 최대한 제공해야 합니다.
강의에 포함되는 각 정보에 대해, 당신은 $20의 팁을 받게 됩니다.
강의에서는 모든 낯선 금융 용어나 개념이 명확하게 설명되어야 하며, 학생들이 해당 주제에 대한 사전 지식이 없다고 가정합니다.
강의에서 강사는 불필요한 반복을 피하고, 논리적인 흐름을 유지해야 합니다.
강의는 마크다운 형식으로 되어야 합니다.
강의는 <lecture></lecture> 태그로 둘러싸여야 합니다.
"""
    return prompt


def input_finance_article(passage):
    prompt = f"""
다음 금융 관련 글을 고려해주세요: <passage>{passage}</passage>
전문적인 금융 기사를 쓰는 기자라고 상상해보세요.
두 가지 작업이 있습니다:
1) 글의 내용에서 영감을 얻어 새로운 금융 기사의 제목을 작성하세요.
기사는 글에서 핵심이 되는 금융 토픽을 응용하여 실생활에서 접할 수 있는 주제로 작성되어야 합니다.
기사 주제는 <title></title> 태그로 둘러싸여야 합니다.
2) 글의 내용을 참고하여 새로운 금융 기사를 작성하세요.
기사는 금융을 잘 모르는 일반인도 이해할 수 있도록 글에서 자세히 설명되지 않은 금융 용어들도 최대한 자세히 설명해야 합니다.
특히 금융 용어를 이해하는데 도움이 되는 예시가 최대한 많이 제공되어야 합니다.
기사는 <article></article> 태그로 둘러싸여야 합니다.
"""
    return prompt


def input_finance_wikipedia(passage):
    prompt = f"""
다음 금융 관련 글을 고려해주세요: <passage>{passage}</passage>
전문적인 금융 회계 분야의 전문가라고 상상해보세요.
한 가지 작업이 있습니다:
글의 내용에서 확인할 수 있는 모든 금융 회계 관련 용어들을 아래의 조건에 맞는 하나의 정돈된 문서로 작성하세요.
용어의 정의는 반드시 포함되어야 합니다.
비율 등 수식으로 정의되거나 계산이 필요한 금융 회계 용어의 경우, 해당 용어가 실제 금융 시장에서 어떤 식으로 활용되는지 배울 수 있는 계산 예시를 반드시 포함해야 합니다.
예시에 대한 풀이도 반드시 포함되어야 합니다. 풀이는 금융을 잘 모르는 일반인도 이해할 수 있도록 단계별로 상세히 작성해야 합니다.
문서는 <document></document> 태그로 둘러싸여야 합니다.
"""
    return prompt