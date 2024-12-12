# ---INFO-----------------------------------------------------------------------

"""
Programs, guidance or otherwise.
"""


# ---DEPEDENCIES----------------------------------------------------------------
import math
import copy
import random
import openai
import guidance

from typing import List, Dict, Optional
from guidance import (
    gen,
    select,
    user,
    silent,
    system,
    assistant,
)
from guidance.models import Model
from guidance._grammar import RawFunction
from .settings import OpenAISettings
from .prompts import (
    hyde_sys_prompt,
    hyde_proceed_prompt,
    frame_fill_sys_prompt,
    frame_fill_user_prompt,
)


# ------------------------------------------------------------------------------
openai_settings = OpenAISettings()


# ---HyDE-----------------------------------------------------------------------
@guidance
def hyde_g(
    lm: Model,
    query: str,
    addendum: Optional[str] = None,
    max_tokens: int = 50,
    temperature: float = 0.1,
    is_chat: bool = True,
) -> Model:
    """
    Program to generate a proxy answer for a query, supporting both chat and
    completion-based models.

    Args:
    lm (Model): The language model.
    query (str): The query.
    addendum (str): Extra considerations.
    max_tokens (int): The maximum number of tokens.
    temperature (float): Sampling temperature.
    is_chat (bool): Whether the model is chat-based.

    Returns:
    lm_hyde (Model): Model with the generated proxy answer.
    """
    sys_prompt = hyde_sys_prompt.format(max_words=int(max_tokens * 0.75))
    proceed_prompt = hyde_proceed_prompt
    if addendum:
        sys_prompt += f"\n\nExtra concerns: {addendum}"
    if is_chat:
        with silent():
            with system():
                lm_hyde = lm + sys_prompt
        with user():
            lm_hyde += f"\n\nQuery: {query}\n"
            with silent():
                lm_hyde += proceed_prompt
        with silent():
            with assistant():
                try:
                    lm_hyde += select(["YES", "NO"], name="hyde")
                except:
                    lm_hyde += gen("hyde", max_tokens=2)
        if lm_hyde["hyde"] == "NO":
            return lm_hyde
        with assistant():
            lm_hyde += gen(
                "hyde",
                max_tokens=max_tokens,
                temperature=temperature,
            )
    else:
        with silent():
            lm_hyde = lm + sys_prompt
        lm_hyde += f"\n\nQuery: {query}\n"
        with silent():
            lm_hyde += proceed_prompt
        if lm_hyde["hyde"] == "NO":
            return lm_hyde
        lm_hyde += gen("hyde", max_tokens=max_tokens, temperature=temperature)
    return lm_hyde


def hyde_openai_chat(
    query: str,
    addendum: str = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 50,
    temperature: float = 0.1,
) -> str:
    """
    Program to generate a proxy answer for a query using OpenAI's chat-based
    models directly. Upto 8x faster than guidance based hyde.

    Args:
    query (str): The query.
    addendum (str): Extra considerations.
    model (str): The model to use.
    max_tokens (int): The maximum number of tokens.
    temperature (float): Sampling temperature.

    Returns:
    response (str): The generated proxy answer.
    """
    sys_prompt = hyde_sys_prompt.format(max_words=int(max_tokens * 0.75))
    proceed_prompt = hyde_proceed_prompt
    if addendum:
        sys_prompt += f"\n\nExtra concerns: {addendum}"
    client = openai.OpenAI(api_key=openai_settings.api_key)
    # Avoiding 2nd API call saves ~0.7s

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {query}\n{proceed_prompt}"},
            {"role": "user", "content": "If YES, then proceed with the answer"},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response = response.choices[0].message.content
    return "NO" if "NO" in response else response


# ---MCTSwRAG-------------------------------------------------------------------
class MCTSReasoningNode:
    """
    Node for the reasoning tree.
    """

    def __init__(
        self,
        lm: Model,
        state: str,
        action: Optional[str] = None,
        parent: Optional["MCTSReasoningNode"] = None,
        children: Optional[List["MCTSReasoningNode"]] = None,
        visits: int = 0,
        Q: float = 0.0,
        exploration_constant: float = 1.414,
        is_terminal: bool = False,
        seen_actions: Optional[Dict[str, RawFunction]] = None,
    ):
        self.lm = lm
        self.state = state
        self.action = action
        self.parent = parent
        if parent is None:
            with user():
                self.lm += f"""\
                ROOT: {state}
                """
            self.lm = self.lm.set("ROOT", state)
        self.children = children if children is not None else []
        self.visits = visits
        self.Q = Q
        self.exploration_constant = exploration_constant
        self.is_terminal = is_terminal
        self.seen_actions = seen_actions if seen_actions is not None else {}

    def __add__(self, prog_mctsa: RawFunction) -> "MCTSReasoningNode":
        if self.is_terminal:
            return self
        child_lm = self.lm + prog_mctsa
        is_terminal = child_lm["is_terminal"] == "YES"
        action_desc = child_lm["action"]
        if action_desc not in self.seen_actions:
            self.seen_actions[action_desc] = prog_mctsa
        else:
            # Avoiding duplicate actions
            return self
        child = MCTSReasoningNode(
            child_lm,
            child_lm["state"],
            child_lm["action"],
            self,
            is_terminal=is_terminal,
            seen_actions=self.seen_actions,
        )
        self.children.append(child)
        return child

    def select(
        self, temperature: float = 0.1, next: bool = False
    ) -> "MCTSReasoningNode":
        if self.is_terminal:
            return self
        node = self
        while len(node.children) > 0:
            w = [child.ucb for child in node.children]
            w = self.softmax(w, temperature)
            node = random.choices(node.children, weights=w, k=1)[0]
            if next:
                break
        return node

    def simulate(self, n_simulations=2, max_rollouts=8, temperature=0.1):
        # TODO: Support for terminal rollouts
        node_copy = self.copy()
        mcts_sim_ans = []
        for _ in range(n_simulations):
            node = node_copy
            for _ in range(max_rollouts):
                if node.is_terminal:
                    break
                action = random.choice(list(node.seen_actions.values()))
                node = node + action
            with user():
                lm_a = (
                    node.lm
                    + f"""\
                ROOT: {node["ROOT"]}
                
                Based on the current chain of thought, try to resolve root. If
                already resolved rewrite the resolution.
                
                Please respond in a 100 tokens or less.
                """
                )
            with assistant():
                lm_a += gen(
                    "mcts_simulate",
                    max_tokens=100,
                    temperature=temperature,
                )
            mcts_sim_ans.append(lm_a["mcts_simulate"])
        with user():
            lm_compare = (
                node_copy.lm
                + f"""\
                Group responses by the specifics of their content. Reponses 
                having similar content, for example, same determiners, same 
                values, same outcomes, etc., should be grouped together. 
                
                Responses:
                """
            )
            for i, ans in enumerate(mcts_sim_ans):
                lm_compare += f"{i + 1}. {ans}\n"
        with assistant():
            lm_compare += gen(max_tokens=110 * n_simulations)
        with user():
            lm_compare += f"""\
            Give me the number of members in the largest group of similar 
            responses.
            
            Write a single number between 1 and {n_simulations}.
            """
        with assistant():
            lm_compare += select(list(range(1, n_simulations + 1)), "major_num")
        reward = lm_compare["major_num"] / n_simulations
        self.update(reward)

    def update(self, reward: float) -> None:
        self.visits += 1
        self.Q += (reward - self.Q) / self.visits
        if self.parent is not None:
            self.parent.update(reward)

    @property
    def ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.Q / self.visits + self.exploration_constant * math.sqrt(
            math.log(self.parent.visits) / (self.visits + 1e-6)
        )

    @staticmethod
    def softmax(values: List[float], temperature=0.1) -> List[float]:
        values = [math.exp(v / temperature) for v in values]
        values = [v / sum(values) for v in values]
        return values

    def copy(self):
        return copy.deepcopy(self)


def mcts_action(fn) -> RawFunction:
    """
    A decorator to replace @guidance for programs intended to be used as actions
    in MCTS. It handles setting the necessary 'action' and 'is_terminal'
    variables during node formation.
    """

    @guidance(dedent=False)
    def wrapper(lm, *args, **kwargs):
        action_desc = f"{fn.__name__}|"
        for i, arg in enumerate(args):
            action_desc += f"arg{i + 1}: {arg}|"
        for key in sorted(kwargs.keys()):
            action_desc += f"{key}: {kwargs[key]}|"
        lm = lm.set("action", action_desc)
        lm = fn(lm, *args, **kwargs)
        lm_temp = lm.copy()
        with user():
            lm_temp += f"""\
            ROOT: {lm_temp["ROOT"]}
            
            Based on the current chain of thought, can you resolve ROOT? Respond 
            with YES if you are certain, otherwise say NO, we can always think 
            more. Please answer with a single word: YES or NO.
            """
        with assistant():
            lm_temp += select(["YES", "NO"], "is_terminal")
        lm = lm.set("is_terminal", lm_temp["is_terminal"])
        return lm

    return wrapper


# ---rSTAR----------------------------------------------------------------------
@mcts_action
def propose_one_step_thought(lm: Model) -> Model:
    with user():
        lm += "Generate the next reasoning step based on previous steps."
    with assistant():
        lm += gen("state", max_tokens=300)
    return lm


@mcts_action
def propose_remaining_thought_steps(lm: Model) -> Model:
    with user():
        lm += "Produce all remaining reasoning steps."
    with assistant():
        lm += gen("state", max_tokens=300)
    return lm


@mcts_action
def generate_next_sub_question_and_answer(lm: Model) -> Model:
    with user():
        lm += "Decompose the main problem into a sequence of sub-questions."
    with assistant():
        lm += gen("state", max_tokens=300)
    return lm


@mcts_action
def re_answer_sub_question(lm: Model) -> Model:
    with user():
        lm += "Re-answer a previously generated sub-question."
    with assistant():
        lm += gen("state", max_tokens=300)
    return lm


@mcts_action
def rephrase_question_sub_question(lm: Model) -> Model:
    with user():
        lm += """\
        Rephrase the question to clarify conditions and reduce 
        misunderstandings.
        """
    with assistant():
        lm += gen("state", max_tokens=300)
    return lm


@guidance
def rstar_g(
    lm: Model,
    root_state: str,
    max_iterations: int = 10,
    max_expansions: int = 2,
    n_simulations: int = 2,
    max_rollouts: int = 8,
    sel_temp: float = 0.1,
    sim_temp: float = 0.1,
    capture_cot_in_state: bool = False,
) -> Model:
    root = MCTSReasoningNode(lm, root_state)
    for _ in range(max_iterations):
        # Selection
        node = root.select(sel_temp)
        # Expansion
        acs = [
            propose_one_step_thought(),
            propose_remaining_thought_steps(),
            generate_next_sub_question_and_answer(),
            re_answer_sub_question(),
            rephrase_question_sub_question(),
        ]
        acs = random.choices(acs, k=min(max_expansions, len(acs)))
        for ac in acs:
            ch = node + ac
        # Simulation
        # Select a random child node for simulation
        node = node.select(next=True)
        # Simulate
        node.simulate(n_simulations, max_rollouts, sim_temp)
        # Backpropagation
        # if terminal, backpropagate reward 1.0
        if node.is_terminal:
            node.update(1.0)
    best_child = max(root.children, key=lambda child: child.visits)
    with user():
        lm_rstar = (
            best_child.lm
            + f"""\
        Resolve ROOT based on the reasoning steps. Rewrite the resolution if
        already resolved.
        """
        )
    with assistant():
        lm_rstar += gen("rstar", max_tokens=300)
    if capture_cot_in_state:
        return lm_rstar
    else:
        with user():
            lm_rstar_nccs = (
                lm
                + f"""\
            ROOT: {lm_rstar["ROOT"]}
            """
            )
        with assistant():
            lm_rstar_nccs += lm_rstar["rstar"]
        return lm_rstar_nccs


# ---Form Filling---------------------------------------------------------------
@guidance
def frame_fill_g(
    lm: Model,
    template: str,
    instructions: str,
    extra_context: Optional[str] = None,
    addendum: Optional[str] = None,
    is_chat: bool = False,
) -> Model:
    """
    Program to fill a form based on a template.

    Args:
    lm (Model): The language model.
    template (str): The template.
    addendum (str): Extra considerations.

    Returns:
    lm_filled (Model): Model with the filled form.
    """
    sys_prompt = frame_fill_sys_prompt
    if addendum:
        sys_prompt += f"\n\nExtra concerns: {addendum}"
    if is_chat:
        with silent():
            with system():
                lm_filled = lm + sys_prompt
        with user():
            lm_filled += frame_fill_user_prompt.format(
                instructions=instructions, extra_context=extra_context
            )
        with assistant():
            lm_filled += template
    else:
        with silent():
            lm_filled = (
                lm
                + sys_prompt
                + frame_fill_user_prompt.format(
                    extra_context=extra_context,
                    instructions=instructions,
                )
            )
        lm_filled += template
    return lm_filled
