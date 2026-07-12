import random
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

module_path = os.path.abspath(os.path.join('src'))

if module_path not in sys.path:
    sys.path.append(module_path)

from DiagnosisTool.pipelines import build_diabetes_pipeline, build_breast_cancer_pipeline
from DiagnosisTool.data import load_diabetes_data, load_breast_cancer_data
from DiagnosisTool.features import clean_diabetes_data, split_diabetes_data, clean_breast_cancer_data, split_breast_cancer_data

from DiagnosisTool.utils import JSONLogger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


logger = JSONLogger(log_file="../src/DiagnosisTool/optimization/results/logs.jsonl")

class GeneticOptimizer:
    def __init__(self, model_builder, X, y, population_size=10, generations=5, mutation_rate=0.3, dataset="unknown"):
        self.model_builder = model_builder
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.param_space = {}
        self.dataset = dataset

        logger.log("genetic_optimizer", "INFO", "optimizer_initialized",
                   details={"dataset": dataset, "population_size": population_size, "generations": generations})

    def initialize_population(self, param_space):
        self.param_space = param_space
        self.population = []
        for _ in range(self.population_size):
            individual = {param: random.choice(values) for param, values in param_space.items()}
            self.population.append(individual)
        logger.log("genetic_optimizer", "INFO", "population_initialized",
                   details={"dataset": self.dataset, "population_size": self.population_size,
                            "initial_population": self.population})

    def fitness(self, individual):
        pipeline = self.model_builder(**individual)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        logger.log("genetic_optimizer", "INFO", "individual_evaluated",
                   details={"dataset": self.dataset, "params": individual,
                            "accuracy": acc, "f1": f1, "recall": rec})

        return (acc + f1 + rec) / 3

    def selection(self):
        scored = [(self.fitness(ind), ind) for ind in self.population]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ind for _, ind in scored[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        child = {}
        for param in self.param_space.keys():
            child[param] = random.choice([parent1[param], parent2[param]])
        return child

    def mutation(self, individual):
        for param, values in self.param_space.items():
            if random.random() < self.mutation_rate:
                individual[param] = random.choice(values)
        return individual

    def evolve(self):
        for gen in range(self.generations):
            selected = self.selection()
            children = []
            while len(children) < self.population_size:
                p1, p2 = random.sample(selected, 2)
                child = self.crossover(p1, p2)
                child = self.mutation(child)
                children.append(child)
            self.population = children
            best = max(self.population, key=lambda ind: self.fitness(ind))
            logger.log("genetic_optimizer", "INFO", "generation_completed",
                       details={"dataset": self.dataset, "generation": gen+1,
                                "best_fitness": self.fitness(best),
                                "best_params": best,
                                "population": self.population})
        logger.log("genetic_optimizer", "INFO", "optimization_completed",
                   details={"dataset": self.dataset, "best_params": best})
        return best

if __name__ == "__main__":
    # Diabetes
    data = load_diabetes_data()
    data = clean_diabetes_data(data)
    X, y = split_diabetes_data(data, "Outcome")

    param_space_diabetes = {
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "C": [0.01, 0.1, 1, 10, 50],
        "solver": ["saga"],
        "max_iter": [100, 200, 300, 500]
    }

    optimizer = GeneticOptimizer(build_diabetes_pipeline, X, y, population_size=8, generations=10, mutation_rate=0.4, dataset="diabetes")
    optimizer.initialize_population(param_space_diabetes)
    best_params = optimizer.evolve()
    print("Melhores hiperparâmetros encontrados (Diabetes):", best_params)

    # Câncer de Mama
    data_bc = load_breast_cancer_data()
    data_bc = clean_breast_cancer_data(data_bc)
    X_bc, y_bc = split_breast_cancer_data(data_bc, "diagnosis")

    param_space_cancer = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.01, 0.1, 1, 5],
        "max_iter": [2000, 5000, 10000]
    }

    optimizer_bc = GeneticOptimizer(build_breast_cancer_pipeline, X_bc, y_bc, population_size=8, generations=10, mutation_rate=0.4, dataset="breast_cancer")
    optimizer_bc.initialize_population(param_space_cancer)
    best_params_bc = optimizer_bc.evolve()
    print("Melhores hiperparâmetros encontrados (Câncer de Mama):", best_params_bc)
