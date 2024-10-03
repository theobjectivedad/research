"""
Copyright 2024 theobjectivedad@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import truncnorm

from sandbox.llm_simulator import LLMSimulatorBase, LLMSimulatorLite


class Evaluate:
    """
    MK2 sandbox implementation of the TextEvolve Evaluate function.

    Technical paper: https://bit.ly/3MoNl7A

    Note:
        Changes from MK1

        * Fixed a bug in the normalized score calculation
        * Added parallelized batch evaluation support
        * Added simulated memory selection method
        * Added debate history selection
        * Integrated the LLMSimulator for generating debate turn prompts
        * Performed code quality checks and improvements
        * Added minor documentation improvements
        * Added implementation notes where appropriate
        * Removed pointless debugging print statements

    """

    def __init__(
        self,
        a: List[str],
        w: np.ndarray,
        r: int,
        c: float,
        l: int,  # noqa: E741
        llm_sim: LLMSimulatorBase | None = None,
    ):
        """
        Initialize the evaluator with the given configuration settings.

        Args:
            a (List[str]): List of debater agents.
            w (np.ndarray): Weight vector for score components.
            r (int): Number of debate rounds.
            c (float): Convergence threshold for early stopping.
            l (int): Debate history parameter (currently a placeholder).
        """
        self.a = a
        self.w = w
        self.r = r
        self.c = c
        self.l = l  # noqa: E741

        # Provide an instance of LLMSimulator, this is used to simulate calling the LLM
        # for demonstrations, unit testing, debugging
        self.llm_sim = llm_sim if llm_sim is not None else LLMSimulatorLite()

    # pylint: disable=unused-argument
    def _debate_turn(
        self,
        x: str,
        y: List[str],
        xi: List[str],
        m_i: List[str],
        i: int,
        j: int,
        k: int,
        round_num: int,
        x_name: str = "input",
        y_name: str = "output",
    ) -> np.ndarray:
        """
        Simulate the LLM scoring process for a debate turn using a truncated normal
        distribution. This simulation controls the coefficient of variation (CV) to
        mimic agents reaching consensus over time.

        Note:
            Actual implementations of this function should check to ensure the
            score values returned from the LLM fall within the 0.0 to 10.0 range.
            Problematic responses can either be retried, dropped, or a scaling heuristic
            can be applied to bring them into the valid range.

        Note:
            Actual implementation should consider error handling and retey logic for
            failures, in particular parse errors or scores that are out of bounds.

        Args:
            x (str): The input context in natural language.
            y (List[str]): List of candidate responses.
            xi (List[str]): Debate history.
            m_i (List[str]): Memories for the current agent.
            i (int): Agent identifier.
            j (int): Number of candidate responses.
            k (int): Number of score components.
            round_num (int): The current round number.
            x_name (str): Name of the input context, used to build the debate turn
            prompt
            y_name (str): Name of the candidate responses, used to build the debate turn
            prompt.

        Returns:
            np.ndarray: A j x k matrix with scores between 0.0 and 10.0.
        """

        # Sample debate turn prompt, this will likely need to be optimized for your use
        # case. The instructions are tuned to be flexible enough to evaluate a broad
        # range of x and y variations and agent personas, while pinning the agent to the
        # evaluation criteria to be externally weighted by confidence, relevancy,
        # accuracy, completness, timeliness, and overall score components.
        prompt_template = "\n".join(
            [
                "Your are the {role}. {role_description}",
                "",
                "You are participating in a multi-agent debate to evaluate the {x_name} against each {y_name} below.",
                "",
                "----- BEGIN {x_name_upper} -----",
                "{x}",
                "----- END {x_name_upper} -----",
                "",
                "----- BEGIN {y_name_upper} TO EVALUATE -----",
                "{y}",
                "----- END {y_name_upper} TO EVALUATE -----",
                "",
                "----- BEGIN DEBATE HISTORY -----",
                "{xi}",
                "----- END DEBATE HISTORY -----",
                "",
                "As the {role}, you recall the following memories relevant to this debate:",
                "",
                "----- BEGIN {role_upper} MEMORIES -----",
                "{m_i}",
                "----- END {role_upper} MEMORIES -----",
                "",
                "Your tasks as the {role}, is to evaluate each {y_name} and contribute to the debate.",
                "Consider the following for each {y_name}:",
                "1. As needed, address previous points raised in the debate as they pertain to your evaluation",
                "2. Analyze the relevance and accuracy for each {y_name}",
                "3. Analyze the completeness and depth for each {y_name}",
                "4. Any potential limitations for each {y_name}",
                "5. Analyze any potential limitations for each {y_name}",
                "6. Analyze any timeliness considerations for each {y_name}, is it currently: {current_time}",
                "7. Missing information and/or context for each {y_name}",
                "",
                "Provide your evaluation in a clear, to-the-point, and concise manner, staying true to your role.",
                "",
                "You must evaluate all {y_count} {y_name}.",
            ]
        )

        agent_profiles: Dict[str, str] = {
            "Critic": ""
            "You are a critical thinker who analyzes responses for flaws and "
            "inconsistencies. You question assumptions, point out logical "
            "fallacies, and identify areas where the responses may be lacking or "
            "misleading.",
            "Supporter": ""
            "You are an advocate who looks for strengths and positive aspects in the "
            "responses. Your highlight the merits of each response, explain "
            "potential benefits, and defend good ideas against criticism.",
            "Neutral Observer": ""
            "You are an impartial observer who balances different viewpoints and "
            "provides objective analysis. You consider all perspectives, "
            "weigh the pros and cons, and offer a balanced evaluation of the "
            "responses.",
        }

        # For the simulation, randomly select an agent role
        role, role_description = random.choice(list(agent_profiles.items()))

        rendered_prompt = self.llm_sim.render_template(
            template=prompt_template,
            template_values={
                "role": role,
                "role_upper": role.upper(),
                "role_description": role_description,
                "x": x,
                "x_name": x_name,
                "x_name_upper": x_name.upper(),
                "y": "\n\n".join(
                    [
                        f"{y_name} {y_index+1}:\n{y_value}"
                        for y_index, y_value in enumerate(y)
                    ]
                ),
                "y_name": y_name,
                "y_name_upper": y_name.upper(),
                "y_count": len(y),
                "xi": (
                    "\n".join(xi) if len(xi) > 0 else "(No debate history, first round)"
                ),
                "m_i": (
                    "\n".join(m_i)
                    if len(m_i) > 0
                    else "(No external memories recalled)"
                ),
                "current_time": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
        )

        self.llm_sim(
            rendered_prompt,
            n=1,
            prefix="EVALUATE_TURN",
            min_tokens=200,
            max_tokens=1500,
            temperature=1.0,
            top_p=1.0,
        )

        # Simulation configuration: The Initial coefficient of variation for the first
        # round.
        initial_cv: float = 0.2

        # Simulation configuration: Percentage by which CV decreases each round.
        cv_decrease_percent: float = 10.0

        # Generate mean scores for each candidate response and score component
        # The mean is randomly chosen between 4.5 and 5.5 to center the scores around
        # the middle of the 0-10 range
        mean_score = np.random.uniform(4.5, 6.5, (j, k))  # Shape: (j, k)

        # Compute the current coefficient of variation (CV) for this round
        # CV decreases with each round by the specified percentage to simulate agents
        # reaching consensus
        current_cv = initial_cv * (1 - cv_decrease_percent / 100.0) ** round_num

        # Standard deviation is calculated as a proportion of the mean score, based on
        # the current CV
        std_dev = mean_score * current_cv  # Shape: (j, k)

        # Set the lower and upper bounds for the scores to ensure they stay within the
        # valid range [0.0, 10.0]
        lower, upper = 0.0, 10.0

        # Generate scores using a truncated normal distribution The scores are generated
        # such that they fall within the [0.0, 10.0] range, following a normal
        # distribution centered on mean_score with std_dev
        scores = truncnorm(
            (lower - mean_score) / std_dev,  # Lower bound in standardized units
            (upper - mean_score) / std_dev,  # Upper bound in standardized units
            loc=mean_score,  # Mean of the distribution
            scale=std_dev,  # Standard deviation of the distribution
        ).rvs()  # Generate random variates

        return scores

    @staticmethod
    def _compute_cv(matrix: np.ndarray) -> float:
        """
        Compute the coefficient of variation (CV) for a given score matrix.

        Args:
            matrix (np.ndarray): A matrix of scores (i x j x k) from which CV is
            computed.

        Returns:
            float: The coefficient of variation of the scores.
        """
        return np.std(matrix) / np.mean(matrix)

    def evaluate_batch(
        self,
        batch: List[Tuple[str, List[str]]],
        x_name: str = "input",
        y_name: str = "output",
        workers: int = 10,
        thread_name_prefix: str = "evaluate_pool",
    ) -> List[Tuple[np.ndarray, List[str]] | None]:
        """
        Evaluates a batch of records in parallel using multiple worker threads.

        This method will always preserve the order of the input batch when returning.

        Args:
            batch (List[Tuple[str, List[str]]]): The batch of records to evaluate. Each
                record is a tuple containing an input string and a list of strings. workers
                (int, optional): The number of worker threads to use. Defaults to 10.
            thread_name_prefix (str, optional): The prefix for the worker thread names.
                Defaults to "evaluate_pool".
            workers (int, optional): The number of worker threads to use. Defaults to 10.
            x_name (str): Name of the input context, used to build the debate turn
                prompt.
            y_name (str): Name of the candidate responses, used to build the debate turn
                prompt.

        Returns:
            List[Tuple[np.ndarray, List[str]] | None]: A list of tuples containing the
            evaluation result (S) and the debate for each record in the batch. If an
            error occurs during evaluation, the tuple will contain None values.

        Raises:
            None

        Examples:
            # Example usage
            batch = [
                ("input1", ["string1", "string2"]),
                ("input2", ["string3", "string4"]),
                ("input3", ["string5", "string6"]),
            ]
            results = evaluate_batch(batch, workers=5)

        """
        pool = ThreadPoolExecutor(
            thread_name_prefix=thread_name_prefix, max_workers=workers
        )

        def worker(
            key: int,
            x: str,
            y: List[str],
            x_name: str,
            y_name: str,
        ) -> Tuple[int, np.ndarray, List[str]] | Tuple[int, None, None]:
            """
            Worker thread inner function that evaluates a batch record.

            Args:
                key (int): The key for the batch record.
                x (str): The input string.
                y (List[str]): The list of strings.


            Returns:
                Tuple[int, np.ndarray, List[str]] | Tuple[int, None, None]: A tuple
                containing the key, the evaluation result (S), and the debate. If an
                error occurs during evaluation, returns a tuple with None values.
            """
            try:
                S, debate = self.evaluate(x=x, y=y, x_name=x_name, y_name=y_name)
                return (key, S, debate)
            except:

                LOG.exception("Error evaluating batch record: %i", key)
                return (key, None, None)

        try:
            # Submit worker pool tasks
            futures = []
            for i, batch_record in enumerate(batch):
                x, y = batch_record
                future = pool.submit(worker, i, x, y, x_name, y_name)
                futures.append(future)

            # Get results in the order of the batch
            keyed_results = []
            for future in as_completed(futures):
                result = future.result()
                keyed_results.append(result)

            # Sort the results based on the original order of the batch
            keyed_results = sorted(keyed_results, key=lambda x: x[0])

            # Remove key from results since it was only needed to maintain order of the
            # final result.
            eval_results: List[Tuple[np.ndarray, List[str]] | None] = [None] * len(
                batch
            )
            for cur_keyed_result in keyed_results:

                key = cur_keyed_result[0]
                S = cur_keyed_result[1]
                debate = cur_keyed_result[2]

                # Skip 'None' values, these are errors
                if S is None or debate is None:
                    continue

                # Else, store the result to be returned
                eval_results[key] = (
                    S,
                    debate,
                )

            return eval_results
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _m(self, i: int, x: str, y: List[str], xi: List[str]) -> List[str]:
        """Placeholder method that simulates retrieving memories from an agent.

        Args:
            i (int): The agent index.
            x (str): Input context in natural language.
            y (List[str]): List of candidate responses.
            xi (List[str]): Debate history, use by the memory retrievela algorithm.

        Returns:
            List[str]: A list of retrieved memories for the agent.
        """

        return [f"Memory {x+1}" for x in range(10)]

    def evaluate(
        self,
        x: str,
        y: List[str],
        x_name: str = "input",
        y_name: str = "output",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Evaluate the candidate responses using the TextEvolve evaluation function.

        Args:
            x (str): Input context in natural language.
            y (List[str]): List of candidate responses.
            x_name (str): Name of the input context, used to build the debate turn
            prompt.
            y_name (str): Name of the candidate responses, used to build the debate turn
            prompt.

        Returns:
            Tuple[np.ndarray, List[str]]: Tuple containing the scoring tensor S with
            dimensions (r x i x j x k) and simulated debate history.
        """
        j = len(y)  # Number of candidate responses
        k = len(self.w)  # Number of score components
        i = len(self.a)  # Number of agents
        S = np.zeros((self.r, i, j, k))  # Initialize scoring tensor

        debate: List[str] = []

        for round_num in range(self.r):
            for agent_num in range(i):

                # Placeholder for debate history (ξ)
                xi = debate[-self.l * round_num :]

                # Retrieve memories for the current agent
                m_i = self._m(i=agent_num, x=x, y=y, xi=xi)

                S[round_num, agent_num] = self._debate_turn(
                    x=x,
                    y=y,
                    xi=xi,
                    m_i=m_i,
                    i=agent_num,
                    j=j,
                    k=k,
                    round_num=round_num,
                    x_name=x_name,
                    y_name=y_name,
                )

                # Simulate a debate history entry for this turn
                ts = int(datetime.now().timestamp() * 1000) % 100000
                debate.append(
                    f"{ts}: round {round_num}, agent {agent_num}, response candidates "
                    f"1 to {j} {{DEBATE_TURN}}"
                )

            # After each round, compute CV and check for early stopping
            round_cv = self._compute_cv(matrix=S[round_num])

            if round_cv <= self.c:
                LOG.info(
                    "Early stopping triggered at Round %i (CV <= %f)",
                    round_num + 1,
                    self.c,
                )

                S = S[: round_num + 1]  # Truncate the tensor to the completed rounds
                break

        return (S, debate)

    def compute_normalized_scores(self, S: np.ndarray) -> np.ndarray:
        """
        Compute the normalized scores for each response candidate.

        Args:
            S (np.ndarray): The scoring tensor with dimensions (r x i x j x k).

        Returns:
            np.ndarray: Normalized score vector for each candidate response.
        """
        return np.sum(S * self.w, axis=(0, 1, 3)) / (
            S.shape[0] * S.shape[1] * S.shape[3]
        )

    @staticmethod
    def compute_softmax_scores(s_norm: np.ndarray) -> np.ndarray:
        """
        Compute the softmax scores for each response candidate.

        Args:
            s_norm (np.ndarray): Normalized score vector.

        Returns:
            np.ndarray: Softmax score vector representing probabilities.
        """
        e_x = np.exp(s_norm - np.max(s_norm))  # Subtract max for numerical stability
        return e_x / e_x.sum(axis=0)

    def select_best_candidate(
        self, S: np.ndarray, y: List[str], probabilistic: bool = False
    ) -> str:
        """
        Select the best response candidate based on normalized or softmax scores.

        Args:
            S (np.ndarray): The scoring tensor with dimensions (r x i x j x k).
            y (List[str]): List of candidate responses.
            probabilistic (bool): If True, selects based on softmax scores; otherwise,
            based on normalized scores.

        Returns:
            str: The selected response candidate.
        """
        # Compute normalized scores
        s_norm = self.compute_normalized_scores(S)

        if probabilistic:
            # Compute softmax scores
            s_phi = self.compute_softmax_scores(s_norm)

            # Select the best response candidate probabilistically using softmax scores
            best_index = np.random.choice(np.arange(len(y)), p=s_phi)
        else:

            # Select the best response candidate using normalized scores
            best_index = np.argmax(s_norm)

        return y[best_index]

print("OK")