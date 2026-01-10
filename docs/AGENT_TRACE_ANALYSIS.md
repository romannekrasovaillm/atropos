# Анализ RL-инфраструктуры Atropos: Синтетические агентные трейсы с интерливинговым рассуждением

## Обзор

Atropos представляет собой продвинутую инфраструктуру для reinforcement learning с фокусом на обучение LLM через генерацию и верификацию агентных трейсов. Проект особенно силён в поддержке **interleaved reasoning** (интерливинговое рассуждение) и **multi-turn tool calling** (многоходовое использование инструментов).

---

## 1. Архитектура генерации синтетических трейсов

### 1.1 Базовые структуры данных

**Файл:** `atroposlib/type_definitions.py`

```python
class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: Content  # str | List[ChatCompletionContentPartParam]
    reward: Optional[float]  # Per-message reward для turn-level learning

class AgentStep(TypedDict):
    step: int
    messages: List[Message]
    reward: float

class ScoredDataGroup(TypedDict):
    tokens: List[List[int]]         # Токенизированные последовательности
    masks: List[List[int]]          # Маскирование для loss (только assistant)
    scores: List[float]             # Награды за каждое поколение
    advantages: Optional[List[List[float]]]  # GRPO advantages per-token
    ref_logprobs: Optional[List[List[float]]]
    messages: Optional[List[List[Message]]]  # Полная история сообщений
```

### 1.2 Поток генерации трейсов

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAJECTORY GENERATION FLOW                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│   Dataset    │────▶│   get_next_item  │────▶│   collect_trajectories │
│  (prompts)   │     │   (env method)   │     │   (rollout generation) │
└──────────────┘     └──────────────────┘     └───────────┬────────────┘
                                                          │
                     ┌────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ROLLOUT LOOP (per group_size completions):                             │
│                                                                         │
│  Turn 1: prompt → model → <think>...</think><tool_call>...</tool_call>  │
│                    ↓                                                    │
│  Turn 2: + tool_response → model → <think>...</think><tool_call>...     │
│                    ↓                                                    │
│  Turn N: → final answer with \boxed{...}                                │
└─────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SCORING & VERIFICATION:                                                 │
│  • Turn-level rewards (R^T_τ): tool correctness, format validity        │
│  • Outcome-level rewards (R^O): final answer verification               │
│  • Advantage computation: Â_τ = Â^T_τ + λ·Â^O (MT-GRPO)                │
└─────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TOKENIZATION:                                                           │
│  • tokenize_for_trainer() → tokens + masks                               │
│  • Маскирование non-assistant ролей                                      │
│  • Опционально: train_on_all_assistant_turns для multi-turn              │
└──────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  API SERVER (/scored_data endpoint):                                     │
│  • Queue management с heterogeneous batching                             │
│  • Trainer requests batches via /batch                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Интерливинговое рассуждение (Interleaved Reasoning)

### 2.1 Ключевая среда: `tool_use_interleaved_thinking.py`

Это **инновационная реализация**, позволяющая модели выполнять tool calls **внутри открытого `<think>` блока**:

```
<think>
Мне нужно решить 2+2. Давайте используем калькулятор.
<tool_call>{"name": "calculator", "arguments": {"expr": "2+2"}}</tool_call>
<tool_response>{"value": 4}</tool_response>
Отлично, результат 4. Теперь могу дать финальный ответ.
</think>

Ответ: \boxed{4}
```

**Ключевые особенности:**

| Аспект | Реализация |
|--------|------------|
| Формат | `<think>` → tool_call → tool_response → `</think>` → `\boxed{answer}` |
| Max turns | `MAX_ROLLOUT_TURNS = 3` |
| Token limit | `MAX_REPLY_TOKENS = 2048`, `MAX_GEN_PER_TURN = 512` |
| Tool bonus | `TOOL_USAGE_BONUS = 0.2` за успешное использование |

**Код парсинга (строки 271-309):**

```python
def _extract_last_call(self, chunk: str):
    """Return JSON dict for the last <tool_call>...</tool_call> block."""
    # Сначала ищем complete tool calls
    matches = self._re_last_call.findall(chunk)
    if matches:
        try:
            return json.loads(matches[-1])
        except Exception:
            pass

    # Fallback: incomplete tool calls (missing </tool_call>)
    last_tool_call_pos = chunk.rfind("<tool_call>")
    if last_tool_call_pos != -1:
        json_start = last_tool_call_pos + len("<tool_call>")
        json_text = chunk[json_start:].strip()
        # Brace counting for partial JSON extraction
        ...
```

### 2.2 Execution Loop (строки 640-974)

```python
async def _collect_trajectories_with_execution(self, prompt_msgs, expected):
    """Real interleaved tool execution mode."""

    while not all(done) and turn_idx < max_turns:
        # Build prompts for active rollouts
        for i in range(num_rollouts):
            if not done[i]:
                prompt_txt = self.tokenizer.apply_chat_template(...)
                prompt_txt += assistant_msgs[i]["content"]  # Продолжаем с того места

        # Execute inference (batched identical or heterogeneous)
        if turn_idx == 0:
            replies = await self._batch_identical_prompts(...)
        else:
            replies = await self._batch_heterogeneous_prompts(...)

        # Process each rollout
        for rollout_idx in active_indices:
            raw = assistant_msgs[rollout_idx]["content"]

            if "</think>" in raw:
                # Think closed → check for boxed answer
                boxed = self._boxed_after_think(raw)
                if boxed:
                    done[rollout_idx] = True
            else:
                if self._is_new_tool_call(raw):
                    # Execute tool and append response
                    call_json = self._extract_last_call(raw)
                    result = await self._exec_tool(call_json)
                    assistant_msgs[rollout_idx]["content"] += (
                        f"</tool_call>\n"
                        f"<tool_response>{json.dumps(result)}</tool_response>\n"
                    )
```

---

## 3. MT-GRPO: Turn-Level Credit Assignment

### 3.1 Теоретическая основа

**Файл:** `environments/tool_use_turnlevel_advantage_server.py`

Реализует формулу из статьи "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment" (Zeng et al., 2025):

```
Â_τ = Â^T_τ + λ · Â^O   для всех turns τ = 1, 2, ..., T
```

Где:
- **Â^T_τ** — стандартизированный turn-level advantage (награда за корректность tool call)
- **Â^O** — стандартизированный outcome-level advantage (успех задачи)
- **λ** — коэффициент баланса (`turn_level_advantage_lambda`, default 1.0)

### 3.2 Конфигурация MT-GRPO

```python
class MTGRPOEnvConfig(BaseEnvConfig):
    turn_level_advantage_lambda: float = 1.0   # λ coefficient
    wrong_call_penalty: float = -0.2           # R^T для ошибок
    tool_execution_reward: float = 0.5         # За valid structure
    tool_match_reward: float = 0.5             # За correct content
    summary_reward: float = 1.0                # За narration turn
    validate_think_blocks: bool = True
    generate_all_gpt_turns: bool = True
    max_gen_per_turn: int = 2048
```

### 3.3 Turn-Level Reward Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    TURN-LEVEL REWARD STRUCTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TURN TYPE          │ REWARD COMPONENTS                         │
│  ───────────────────┼───────────────────────────────────────────│
│  Tool Call Turn     │ +0.5 (structure) + 0.5 (content match)    │
│                     │ OR -0.2 (wrong call penalty)              │
│  ───────────────────┼───────────────────────────────────────────│
│  Summary/Narration  │ +1.0 (valid <think> + plain text)         │
│                     │ 0.0 if contains <tool_call>               │
│  ───────────────────┼───────────────────────────────────────────│
│  Final Answer       │ Outcome-level: correct/incorrect          │
│                     │ Verified via math_verify or regex         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Система верификации

### 4.1 Архитектура Reward Functions

**Файл:** `atroposlib/envs/reward_fns/reward_function.py`

```python
class RewardFunction(ABC):
    def __init__(self, weight: float = 1.0, name: Optional[str] = None, **kwargs):
        self.weight = weight
        self._name = name
        self.wandb_logger = None

    @abstractmethod
    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        pass

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        rewards = self.compute(completions, **kwargs)
        weighted_rewards = [r * self.weight for r in rewards]
        return weighted_rewards
```

### 4.2 Иерархия верификаторов

| Reward Function | Файл | Что проверяет |
|-----------------|------|---------------|
| **FormatReasoningReward** | `r1_reward.py` | `<think>...</think>` блоки |
| **AccuracyReward** | `accuracy_reward.py` | `\boxed{answer}` + math_verify |
| **AccuracyXReward** | `r1_reward.py` | Substring matching в response |
| **R1Reward** | `r1_reward.py` | Комбинация format + accuracy |
| **CombinedReward** | `combined_reward.py` | Weighted sum любых rewards |
| **RepetitionPenaltyReward** | `repetition_penalty_reward.py` | Penalty за повторения |

### 4.3 Math Verification Pipeline

**Файл:** `atroposlib/envs/reward_fns/accuracy_reward.py`

```python
def _verify_answer(content: str, gold_answer, tolerance: float = 1e-6) -> bool:
    """Multi-stage verification with fallbacks."""

    # Stage 1: math_verify library (LaTeX parsing + symbolic verification)
    try:
        answer_parsed = parse(
            content,
            extraction_config=[LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    boxed="all",
                    equations=True,
                    units=True,
                ),
                boxed_match_priority=0,
            )],
            extraction_mode="first_match",
        )
        if answer_parsed:
            gold_parsed = parse(f"\\boxed{{{gold_answer}}}", ...)
            return verify(answer_parsed, gold_parsed)
    except Exception:
        pass

    # Stage 2: Regex extraction of \boxed{} content
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", content)
    if boxed_matches and gold_value is not None:
        extracted_value = float(boxed_matches[0].replace(",", ""))
        return abs(extracted_value - gold_value) < tolerance

    # Stage 3: GSM8K style (#### answer)
    if "####" in content:
        match = re.search(r"####\s*([\d\.]+)", content)
        if match:
            return abs(float(match.group(1)) - gold_value) < tolerance

    return False
```

---

## 5. Управление траекториями

### 5.1 Thinking Truncation

**Файл:** `atroposlib/utils/message_history_utils.py`

```python
def truncate_thinking(response_text: str, tokenizer, max_think_tokens: int) -> str:
    """
    Обрезает <think> блок с сохранением семантики:

    1. Если блок <= max_think_tokens → оставляем как есть
    2. Иначе пробуем сохранить последний параграф (если он <= limit)
    3. Fallback: обрезаем с конца и добавляем "..."
    """

    # Парсим think блок
    think_content = response_text[think_start+len("<think>"):think_end]
    all_think_tokens = tokenizer.encode(think_content, add_special_tokens=False)

    if len(all_think_tokens) <= max_think_tokens:
        return response_text  # Не трогаем

    # Пробуем paragraph-based truncation
    paragraphs = [p.strip() for p in think_content.split("\n\n") if p.strip()]
    last_paragraph = paragraphs[-1]
    last_para_tokens = tokenizer.encode(last_paragraph, add_special_tokens=False)

    if len(last_para_tokens) <= max_think_tokens:
        final_think_tokens = last_para_tokens
    else:
        # Fallback: truncate from end
        final_think_tokens = all_think_tokens[-max_think_tokens:]

    decoded = tokenizer.decode(final_think_tokens)
    return f"{part_before}<think>\n... {decoded}\n</think>{part_after}"
```

### 5.2 Trajectory Token Limit

```python
def ensure_trajectory_token_limit(trajectory, tokenizer, max_trajectory_tokens):
    """
    Гарантирует token limit для trajectory:

    Стратегия:
    1. Сохраняем: system prompt, последнее observation, последний response
    2. Удаляем: message пары из старейших шагов (FIFO)
    3. Если всё ещё превышает — дисквалифицируем step
    """

    for step_idx, step_data in enumerate(trajectory):
        max_tokens = max(len(alt) for alt in step_data["tokens"])

        if max_tokens <= max_trajectory_tokens:
            continue  # Compliant

        # Truncation loop
        while max_current_tokens > max_trajectory_tokens:
            # Find what we can pop (preserve last env+agent pair)
            for alt_idx in range(num_alternatives):
                if available_to_pop >= 2:
                    # Pop environment + agent pair
                    working_messages[alt_idx].pop(1)
                    working_messages[alt_idx].pop(1)

            # Re-tokenize and check
            tokenized = tokenize_for_trainer(tokenizer, working_messages[alt_idx])
            max_current_tokens = len(tokenized["tokens"])
```

---

## 6. GRPO Advantage Computation

**Файл:** `atroposlib/utils/advantages.py`

```python
def compute_grpo_process_supervision_advantages(
    rewards: Sequence[Sequence[number]],
    gamma: float = None,
    std_tol: float = 1e-8
) -> list[np.ndarray]:
    """
    GRPO advantage computation для process supervision.

    Args:
        rewards: Jagged list — rewards[trajectory][step]
        gamma: Discount factor (None → cumulative sum)

    Returns:
        Per-step advantages для каждой trajectory
    """

    # Step 1: Global normalization
    stats = compute_stats(rewards)  # mean, var across ALL trajectories
    mean, std = stats["mean"], stats["var"] ** 0.5

    # Step 2: Normalize each trajectory
    normalized = [(np.array(traj) - mean) / std for traj in rewards]

    # Step 3: Compute advantages
    if gamma is None:
        # Cumulative sum (no discounting)
        advantages = [
            np.flip(np.cumsum(np.flip(traj)))
            for traj in normalized
        ]
    else:
        # Discounted returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        advantages = [
            compute_discounted_returns(traj, gamma)
            for traj in normalized
        ]

    return advantages
```

---

## 7. Tokenization с Selective Masking

**Файл:** `atroposlib/utils/tokenize_for_trainer.py`

```python
UNMASKED_ROLES = ["assistant", "agent"]

def tokenize_for_trainer(
    tokenizer,
    chat: list[Message],
    train_on_all_assistant_turns: bool = False,
):
    """
    Два режима маскирования:

    1. train_on_all_assistant_turns=False (default):
       - Маскируем ВСЁ кроме последнего assistant turn
       - masks = [-100] * prefix_len + tokens[prefix_len:]

    2. train_on_all_assistant_turns=True:
       - Маскируем только non-assistant roles
       - Каждый assistant turn получает gradient
    """

    tokens = tokenizer.apply_chat_template(chat)

    if not train_on_all_assistant_turns:
        # Only last turn
        prefix_len = len(tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True))
        masks = [-100] * prefix_len + tokens[prefix_len:]
    else:
        # All assistant turns
        masks = np.ones(len(tokens)) * -100

        for i, msg in enumerate(chat):
            if msg["role"] in UNMASKED_ROLES:
                prefix = tokenizer.apply_chat_template(chat[:i], add_generation_prompt=True)
                unmasked = tokenizer.apply_chat_template(chat[:i+1])
                masks[len(prefix):len(unmasked)] = tokens[len(prefix):len(unmasked)]

    return {"tokens": tokens, "masks": masks.tolist()}
```

---

## 8. Агентные среды в Community

### 8.1 Классификация по типу верификации

| Среда | Тип агента | Верификация |
|-------|------------|-------------|
| `protein_design/` | Multi-tool scientific | AlphaFold2, ProteinMPNN execution |
| `pokemon-showdown/` | Game playing | Battle outcome |
| `playwright_agent_env.py` | Browser automation | Action success + state verification |
| `router_env/` | Multi-agent routing | Agent selection correctness |
| `cybersecurity_sigma/` | Security analysis | Jaccard similarity + LLM judge |
| `physical_space_stl/` | 3D generation | STL rendering + constraint check |

### 8.2 Пример: Protein Design Environment

```python
# environments/community/protein_design/protein_env.py

class ProteinDesignEnv(BaseEnv):
    """
    Multi-turn protein design with tool calling:
    1. RFdiffusion: structure generation
    2. ProteinMPNN: sequence design
    3. AlphaFold2: structure prediction & validation
    """

    async def collect_trajectories(self, item):
        # Model generates tool calls
        # Tools execute with real molecular simulation
        # Verification: pLDDT score, structural metrics
        pass
```

---

## 9. API Server Integration

**Файл:** `atroposlib/api/server.py`

```python
@app.post("/scored_data")
async def scored_data(scored_data: ScoredData):
    """Получает завершённые trajectory данные от окружения."""
    # scored_data содержит:
    # - tokens: List[List[int]]
    # - masks: List[List[int]]
    # - scores: List[float]
    # - advantages: Optional[List[List[float]]]
    # - messages: Optional[List[List[Message]]]

@app.get("/batch")
async def get_batch():
    """Trainer запрашивает batch для обучения."""
    # Использует heterogeneous queue management
    # для multi-environment batching
```

---

## 10. Ключевые выводы

### 10.1 Сильные стороны архитектуры

1. **Flexible Interleaved Reasoning**: Уникальная возможность tool calls внутри `<think>` блоков
2. **Turn-Level Credit Assignment**: MT-GRPO формула для fine-grained learning
3. **Multi-Stage Verification**: Fallback strategies для robust answer checking
4. **Selective Masking**: Гибкий контроль над тем, какие части trajectory получают gradient
5. **Trajectory Management**: Интеллектуальное truncation с сохранением семантики

### 10.2 Возможности для расширения

| Направление | Текущее состояние | Потенциал |
|-------------|-------------------|-----------|
| Tree-of-Thought | Линейные трейсы | Branching verification |
| Self-correction | Implicit в multi-turn | Explicit correction loops |
| Meta-learning | Фиксированные промпты | Dynamic few-shot |
| Multi-agent | Single-agent focus | Agent coordination rewards |

### 10.3 Рекомендации по использованию

1. **Для math reasoning**: Используйте `InterleavedInlineEnv` с `python_interpreter` tool
2. **Для multi-step tasks**: Используйте `MTGRPOEnv` с `turn_level_advantage_lambda=1.0`
3. **Для custom domains**: Наследуйте от `BaseMTGRPOEnv` и override:
   - `compute_turn_reward()` — turn-level signals
   - `compute_outcome_reward()` — final verification
   - `validate_tool_call_turn()` — custom tool parsing

---

## 11. Файловая структура ключевых компонентов

```
atropos/
├── atroposlib/
│   ├── api/
│   │   └── server.py                    # Trajectory API endpoints
│   ├── envs/
│   │   ├── base.py                      # BaseEnv, ScoredDataGroup
│   │   └── reward_fns/
│   │       ├── reward_function.py       # Abstract base class
│   │       ├── accuracy_reward.py       # Math verification
│   │       ├── r1_reward.py             # Format + accuracy
│   │       └── registry.py              # Reward factory
│   ├── utils/
│   │   ├── message_history_utils.py     # Trajectory truncation
│   │   ├── tokenize_for_trainer.py      # Selective masking
│   │   ├── advantages.py                # GRPO computation
│   │   └── tool_call_parser.py          # Tool extraction
│   └── type_definitions.py              # Message, AgentStep, etc.
│
└── environments/
    ├── tool_use_interleaved_thinking.py       # Interleaved reasoning
    ├── tool_use_turnlevel_advantage_server.py # MT-GRPO
    ├── tool_use_multiturn_server.py           # Multi-turn baseline
    ├── gsm8k_server.py                        # Math with CoT
    ├── math_server.py                         # Advanced math
    └── community/                             # 40+ specialized envs
```

---

*Автор: Claude Code Analysis*
*Дата: 2026-01-10*
