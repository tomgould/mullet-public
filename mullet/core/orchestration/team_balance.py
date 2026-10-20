"""Distributes tasks evenly among agents based on current load."""

class TeamLoadBalancer:
    """Assign tasks to the least-loaded agent."""

    def __init__(self):
        self.agent_load = {}

    def assign_task(self, agents, task):
        """Assign a task to the least loaded available agent."""
        if not agents:
            return None

        for agent in agents:
            if agent not in self.agent_load:
                self.agent_load[agent] = 0

        least_loaded_agent = min(agents, key=self.agent_load.get)

        self.agent_load[least_loaded_agent] += 1
        return least_loaded_agent

    def complete_task(self, agent):
        """Decrement the load counter for a finished agent."""
        if agent in self.agent_load and self.agent_load[agent] > 0:
            self.agent_load[agent] -= 1
