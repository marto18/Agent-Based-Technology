
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt

# Матрица на печалбите: T=Изкушение, R=Награда, P=Наказание, S=Печалба на жертвата
T = 3  # Изкушение да предадеш
R = 2  # Награда за взаимно сътрудничество
P = 1  # Наказание за взаимна дефекция
S = 0  # Печалба на жертвата


# Клас Prisoner (Затворник), който наследява Agent от Mesa
class Prisoner(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.strategy = random.choice(['C', 'D'])  # Стратегията е случайно C или D
        self.payoff = 0

    def step(self):
        # Взимаме съседите на агента
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        # Играем дилемата на затворника със всеки съсед
        for neighbor in neighbors:
            my_payoff, neighbor_payoff = self.play_prisoners_dilemma(neighbor)
            self.payoff += my_payoff
            neighbor.payoff += neighbor_payoff

    def play_prisoners_dilemma(self, opponent):
        # Правила за определяне на печалбата според стратегиите на агентите
        if self.strategy == 'C' and opponent.strategy == 'C':
            return R, R
        elif self.strategy == 'C' and opponent.strategy == 'D':
            return S, T
        elif self.strategy == 'D' and opponent.strategy == 'C':
            return T, S
        else:
            return P, P

    def advance(self):
        # Променяме стратегията на агента въз основа на печалбата му
        if self.payoff >= 3:
            self.strategy = 'C'
        else:
            self.strategy = 'D'
        # Нулираме печалбата за следващия кръг
        self.payoff = 0


# Класът PrisonersDilemmaModel, който представлява модела на играта
class PrisonersDilemmaModel(Model):
    def __init__(self, width, height, N):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Създаваме агентите и ги разполагаме на случаен принцип в мрежата
        for i in range(self.num_agents):
            a = Prisoner(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Събираме данни за броя на сътрудничещите агенти (C) и мълчащите агенти (D)
        self.datacollector = DataCollector(
            {
                "Cooperators": lambda m: self.count_type(m, 'C'),
                "Defectors": lambda m: self.count_type(m, 'D')
            }
        )

    def step(self):
        # Събираме данни за текущата стъпка
        self.datacollector.collect(self)
        self.schedule.step()

    @staticmethod
    def count_type(model, strategy):
        count = 0
        for agent in model.schedule.agents:
            if agent.strategy == strategy:
                count += 1
        return count


# Изпълнение на симулацията
model = PrisonersDilemmaModel(10, 10, 50)  # 10x10 мрежа с 50 агенти
for i in range(100):  # Изпълняваме модела за 100 стъпки
    model.step()

# Събиране на данни
data = model.datacollector.get_model_vars_dataframe()
print(data)

# Визуализация на резултатите
plt.figure(figsize=(10, 6))
plt.plot(data["Cooperators"], label="Cooperators (C)", color="green")
plt.plot(data["Defectors"], label="Defectors (D)", color="red")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.title("Prisoner's Dilemma: Cooperators vs. Defectors")
plt.legend()
plt.grid(True)
plt.show()



