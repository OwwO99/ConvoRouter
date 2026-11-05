from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(
            self,
            dataset,
            system_role='assistant',
            user_role='user',
        ):
        self.system_role = system_role
        self.user_role = user_role
        self.dataset = dataset

    @abstractmethod
    def get_init_prompt(self, case):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def get_eval_prompt(self, case, dial):
        raise NotImplementedError("Subclasses must implement this method")
    

class JDDC(Task):
    def __init__(self, dataset, system_role='assistant', user_role='user'):
        super().__init__(
            dataset=dataset, 
            system_role=system_role, 
            user_role=user_role
        )
    
    def get_init_prompt(self, case):
        init_prompt = case['init_prompt']

        rules = case['agent_knowledge_and_rules']['rules']
        knowledges = case['agent_knowledge_and_rules']['prior_knowledge']
        rule = ''
        for i in range(len(rules)):
            rule += f'规则{i}：' + rules[i] + '\n'
        for i in range(len(knowledges)):
            rule += f'信息{i}：' + knowledges[i] + '\n'
        system_sys = """
你是一名的电商平台的客服人员，负责处理用户的订单相关咨询。请根据以下公司规则解答用户问题，并保持友好、专业的态度。

你应知道的公司规则和先验信息：
%s
请你遵守以上内容，不要编造信息。
""" % rule

        intents = case['user_goals']
        intent_info = ''
        for i in range(len(intents)):
            intent_info += "目标%s：%s，%s" % (i + 1, intents[i]['intent'], intents[i]['detailed_description'])
        user_sys = """
你是一名网购平台用户，以下是你的用户背景和目标信息：

【行为特征】
你需要在沟通中表现出以下行为倾向：%s。

【背景情况】
%s。

【目标】
%s
你要模拟一位真实的用户，与你的目标一致，并根据客服回复展开自然的多轮对话。请遵循以下规则：

1. 你只能扮演用户，不要模拟或帮助客服说话。
2. 每一轮对话中，你需要根据客服上一轮的回复内容，合理地生成下一轮用户发言（user_reply），语气应自然、符合身份。
3. 请确保你的每一轮发言都紧扣你的用户目标，不要主动提出与目标无关的问题，也不要转移话题。
4. 你的目的是咨询信息，只要你已经获取到满足目标的完整信息，就可以结束对话，无需等待额外确认或操作。
5. 在你认为所有目标都已完成，或者客服明确表示无法继续处理你的请求时，你的发言应为：<结束对话>
""" % (case['user_profile']['behavior_patterns'], case['user_profile']['general_context'], intent_info)
        
        return init_prompt, system_sys, user_sys
    
    def get_eval_prompt(self, case, dial):
        eval_prompt = """
你是一个专业的客服对话评价器，任务是根据 checklist 对客服在对话中的行为进行逐项打分，并判断对话是否已结束。

=== 对话历史 ===
%s

=== CheckList ===
%s

=== 评分说明 ===
你将依据 checklist 中每一项，判断客服是否完成了该服务动作，并为每项打分：
- 1 表示客服明确且充分完成了该项内容；
- 0 表示客服完全未涉及或明显未完成该项；
- 0.5 表示客服部分完成，但表达模糊、不充分或仅间接涉及。

请严格结合对话内容和 checklist 的实际涵盖情况进行判断，不做主观合理化补全。

=== 终止判断 ===
若且仅若用户在最后一轮的完整回复严格等于字符串"<结束对话>"，则认为对话已结束，done 设为 1，否则为 0。

=== 输出格式要求 ===
仅返回以下 JSON 格式，不要包含其他内容：

{
  "checklist": [打分1, 打分2, ...],
  "done": 0 或 1
}
""" % (dial, case['checklist'])
        return eval_prompt


class MedDG(Task):
    def __init__(self, dataset, system_role='doctor', user_role='patient'):
        super().__init__(
            dataset=dataset, 
            system_role=system_role, 
            user_role=user_role
        )
    
    def get_init_prompt(self, case):
        init_prompt = case['init_prompt']

        system_sys = """
你是一名在线问诊平台的医生，负责接诊患者的健康咨询。请保持专业、耐心、自然的态度，像一位真实医生一样进行多轮问诊、分析和建议。

你应遵守以下规则：

1. 循序问诊：根据患者逐步提供的信息进行综合判断。每轮只需聚焦 1–2 个重点问题或建议，避免一次性讲解过多内容。
2. 主动追问：信息不足时优先补充病史和症状细节，不要提前下结论或全面总结。
3. 自然表达：语气要贴近真实医生的口吻，表达流畅、有交流感，避免使用列表、编号、医学讲解式或书面报告式语言。
4. 简洁聚焦：每次回复保持简短（建议不超过 5 行），内容集中在当前对话核心，不重复解释药物机制或常识。
5. 渐进建议：当信息较充分时，再给出少量核心的检查或用药建议（不超过 2 项），重点说明“为什么”而非全面罗列。
6. 不作臆测：不得虚构医学事实、补全未提供的症状或自行推断检查结果。
7. 语气人性化：关注患者情绪，用自然的安抚、肯定或鼓励性表达，而非模板化安慰语。
8. 避免机械总结：不要重复患者已提供的信息或复述病史，要以医生的思路推进问诊或判断。
"""

        intents = case['user_goals']
        intent_info = ''
        for i in range(len(intents)):
            intent_info += "目标%s：%s" % (i + 1, intents[i]['intent'])
        user_sys = """
你是一位正在进行在线问诊的患者，以下是你的完整背景信息和就诊目标：

【病人信息】
你是%s岁的病人，性别%s，你最近出现了%s，目前正在向医生咨询。你%s。你的既往病史为%s，当前是否用药为%s。

【症状细节】
以下是你当前症状的具体描述：%s

【目标】
%s

1. 你只能扮演“患者本人”，不可模拟医生角色，也不可同时扮演他人（如家属、朋友）。
2. 每一轮你需要根据医生上一轮的回复内容，生成一条自然、真实、有逻辑的发言，用“我”表达。
3. 你应表现出“寻求帮助、等待判断”的自然患者状态，不要纠正、肯定、解释医生的答案。
4. 请确保你的每一轮发言都紧扣你的用户目标，不要主动提出与目标无关的问题，也不要转移话题。
5. 仅当你确信所有目标都已完成，或者医生明确表示无法继续处理你的请求时，才可以结束对话。
    在做出结束判断前，你应检查：
   - 所有目标是否都被医生清晰、准确地回应；
   - 医生的建议是否明确、合理；
   - 没有仍需澄清的疑问或遗漏。
   若不满足以上任一条件，请继续追问，而不要提前使用 <结束对话>。
6. 当你确认对话已满足结束条件时，你的发言应为：<结束对话>
""" % (case['user_profile']['age'], case['user_profile']['gender'], case['user_profile']['chief_complaint'],
       case['user_profile']['suspected_trigger'], case['user_profile']['medical_history'], case['user_profile']['current_medication'],
       case['user_profile']['symptom_details'], intent_info)

        return init_prompt, system_sys, user_sys
    
    def get_eval_prompt(self, case, dial):
        eval_prompt = """
你是一个专业的医疗问诊对话评价器，任务是根据 checklist 对医生在对话中的表现进行逐项打分，并判断问诊是否已结束。

=== 对话历史 ===
%s

=== CheckList ===
%s

=== 评分说明 ===
你将依据 checklist 中的每一项，判断医生是否完成了该项问诊动作，并为每项打分：
- 1 表示医生明确且充分完成了该项内容，信息完整、清晰、具备医学合理性；
- 0 表示医生完全未涉及或明显未履行该项职责；
- 0.5 表示医生部分完成，但表达模糊、信息不足或仅间接回应。

请严格结合实际对话内容与 checklist 项所要求的服务行为进行判断，不得主观推测或合理化补全医生的行为。

=== 终止判断 ===
若且仅若患者在最后一轮的完整发言严格等于字符串 "<结束对话>"，则认为问诊已结束，done 设为 1，否则为 0。

=== 输出格式要求 ===
仅返回以下 JSON 格式，不要包含其他内容：

{
  "checklist": [打分1, 打分2, ...],
  "done": 0 或 1
}
""" % (dial, case['checklist'])
        return eval_prompt
    

class ShareGPT(Task):
    def __init__(self, dataset, system_role='assistant', user_role='user'):
        super().__init__(
            dataset=dataset, 
            system_role=system_role, 
            user_role=user_role
        )

    def get_init_prompt(self, case):
        init_prompt = case['init_prompt']

        system_sys = """
You are an intelligent and helpful AI assistant.  
Your goal is to provide accurate, detailed, and contextually relevant answers to user queries.  

Instructions to strictly follow:  
1. Keep all responses concise, focused, and easy to read. Avoid unnecessary elaboration, repetition, or long-winded explanations.  
2. When explaining complex topics, break them into short, clear steps. Use concise sentences that convey the essential information.  
3. Limit your response to approximately 10 lines or fewer, with each line containing a single idea or step.  
4. Prioritize brevity without losing accuracy or clarity. Only provide content directly relevant to the user's query.  
5. Be polite and professional, and focus solely on helping the user achieve their goal.  

Always avoid extra detail or tangents beyond what is necessary to fully answer the question.
"""

        intents = case['user_goals']
        intent_info = ''
        for i in range(len(intents)):
            intent_info += (
                f"Goal {intents[i]['id']}: {intents[i]['intent']}\n"
                f"Description:\n"
                f"  - If the current context satisfies `{intents[i]['trigger_logic']['initial']}`, trigger this intent and generate the corresponding user utterance;\n"
                f"  - If the intent has been triggered but not yet fulfilled, follow `{intents[i]['trigger_logic']['re_trigger']}` to determine whether to attempt it again;\n"
                f"  - If the current context matches `{intents[i]['trigger_logic']['no_trigger']}`, do not trigger this intent (skip it or continue with the existing task).\n\n"
            )
        user_sys = """
You are a human user. Please engage in natural, multi-turn conversation with the AI assistant.

【Goals】
Throughout the conversation, you should follow the intents below:
%s

1. Each turn must closely follow the triggered intent; do not go off-topic or introduce unrelated goals. Use natural, conversational language and avoid field-style statements.
2. If no intents can be triggered, or all intents have been completed and cannot be triggered again, output `<End of Conversation>` as the mark for this turn and for ending the conversation.
3. Each turn should coherently connect with the assistant's previous reply; you may exhibit realistic user traits (slight misunderstandings, repetition, follow-up questions, polite feedback, etc.).
4. Keep utterances of moderate length. When necessary, you may ask clarifying questions or provide additional information to help accomplish the task.
""" % (intent_info)
        
        return init_prompt, system_sys, user_sys

    def get_eval_prompt(self, case, dial):
        eval_prompt = """
You are a professional dialogue task evaluator. Your task is to score the assistant's performance in the conversation according to the checklist and determine whether the conversation has ended.

=== Dialogue History ===
%s

=== CheckList ===
%s

=== Scoring Instructions ===
For each item in the checklist, determine whether the assistant has completed the corresponding task behavior in the conversation, and assign a score:
- 1 indicates the assistant clearly and fully completed the item, providing complete, clear information in line with the user's intent;
- 0 indicates the assistant did not address the item at all or clearly failed to fulfill it;
- 0.5 indicates the assistant partially completed the item, but the expression is vague, information is insufficient, or it only indirectly responded.

Strictly base your judgment on the actual dialogue content and the required behaviors in the checklist. Do not speculate or rationalize any assistant behavior that did not occur.

=== End-of-Conversation Determination ===
Consider the conversation ended and set done to 1 **only if** the user's last complete utterance is exactly the string "<End of Conversation>"; otherwise, set done to 0.

=== Output Format ===
Return **only** the following JSON format, without any additional content:

{
  "checklist": [score1, score2, ...],
  "done": 0 or 1
}
""" % (dial, case['checklist'])
        return eval_prompt
    

def create_task(dataset, task_name):
    if task_name == 'jddc':
        task = JDDC(
            system_role='assistant',
            user_role='user',
            dataset=dataset,
        )
    elif task_name == 'meddg':
        task = MedDG(
            system_role='doctor',
            user_role='patient',
            dataset=dataset,
        )
    elif task_name == 'sharegpt':
        task = ShareGPT(
            system_role='assistant',
            user_role='user',
            dataset=dataset,
        )
    return task
