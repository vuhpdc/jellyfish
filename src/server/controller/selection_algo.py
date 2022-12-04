import copy
import logging
from src.server.controller.manager import DNNModel
from src.server.controller.utils import MappingInfo
import math
import random
import numpy as np

'''
Helper functions
'''


def printModelIDsOnGpu(modelsInfo):
    sorted_model_ids = [model_id for model_id, model in sorted(
        modelsInfo.items(), reverse=False, key=lambda x: x[1].gpu_number)]
    return sorted_model_ids


class Cache(object):
    def __init__(self):
        self.cache = {}

    def reset(self):
        self.cache = {}

    def read(self, key):
        if key in self.cache:
            return self.cache[key]

        return None

    def update(self, key, value):
        self.cache[key] = value


'''
Algorithm for model adaptation, i.e., upgrade and degrade.
'''


def ModelSelectionSimulatedAnnealing(simData, initialModelsInfo, clientsInfo, MappingAlgo, debug=False,
                                     tempInitial=0.0125, tempMin=0.0005, shuffleModels=False):
    '''
    Simulated Annealing algorithm
    '''
    DEGRADE = 1
    UPGRADE = 2
    THRESHOLD_EFFECTIVENESS = simData.effectiveness_threshold
    TEMP_REDUCTION_RATIO = 0.99

    def adaptModelsInfo(modelsInfo, model_id_lst, action_lst):
        modelsInfo = modelsInfo

        # TODO: can we avoid the deepcopy?
        newModelsInfo = copy.deepcopy(modelsInfo)

        for idx, model_id in enumerate(model_id_lst):
            modelNumber = modelsInfo[model_id].model_number
            gpu_number = modelsInfo[model_id].gpu_number

            newModelNumber = modelNumber + action_lst[idx]
            if newModelNumber >= simData.num_models:
                newModelNumber = simData.num_models - 1
            elif newModelNumber < 0:
                newModelNumber = 0

            if action_lst[idx] != 0:
                del newModelsInfo[model_id]
                newModel = DNNModel(simData,
                                    model_number=newModelNumber, gpu_number=gpu_number)
                assert newModel.id not in newModelsInfo, print(
                    "Key already exists in the mapping info")
                newModelsInfo[newModel.id] = newModel

        # Reset models
        for model in newModelsInfo.values():
            model.reset()

        return newModelsInfo

    def getOrderedModelIDs(modelsInfo, reverse=False):
        # For action DEGRADE, sort models in decreasing order of their modelNumber
        # For action UPGRADE, sort models in ascending order of their modelNumber
        # reverse = True if action == DEGRADE else False
        sorted_model_ids = [model_id for model_id, model in sorted(
            modelsInfo.items(), reverse=reverse, key=lambda x: x[1].model_number)]
        return sorted_model_ids

    def betterMapping(newMappingInfo, bestMappingInfo, action):
        if action == DEGRADE and \
                bestMappingInfo.metrics.effectiveness < newMappingInfo.metrics.effectiveness:
            return True

        if action == UPGRADE:
            effectiveAcc = bestMappingInfo.metrics.accuracy_per_request
            newEffectiveAacc = newMappingInfo.metrics.accuracy_per_request
            if effectiveAcc < newEffectiveAacc and \
                    newMappingInfo.metrics.effectiveness >= THRESHOLD_EFFECTIVENESS:
                return True

        return False

    def getEnergyCost(newMappingInfo, bestMappingInfo, action):
        cost = None
        if action == DEGRADE:
            cost = newMappingInfo.metrics.effectiveness - \
                bestMappingInfo.metrics.effectiveness

        elif action == UPGRADE:
            effectiveAcc = bestMappingInfo.metrics.accuracy_per_request
            newEffectiveAacc = newMappingInfo.metrics.accuracy_per_request

            if (newMappingInfo.metrics.effectiveness >= THRESHOLD_EFFECTIVENESS):
                cost = newEffectiveAacc - effectiveAcc

        return cost

    def acceptNeighbour(newMappingInfo, currMappingInfo, action, T, C=1.0):
        if betterMapping(newMappingInfo, currMappingInfo, action):
            return True

        deltaEnergy = getEnergyCost(newMappingInfo, currMappingInfo, action)

        if deltaEnergy is not None:
            deltaEnergy = abs(deltaEnergy)
            exp_term = -(deltaEnergy * C) / T

            prob = math.exp(exp_term)
            if prob > random.uniform(0, 1):
                return True

        return False

    def stopCondition(mappingInfo, action, temparature, iter, iterWithoutSol):
        # An explicit stop condition only for the degrade action
        if action == DEGRADE and \
                mappingInfo.metrics.effectiveness >= THRESHOLD_EFFECTIVENESS:
            return True

        # No use of upgrade as effectiveness is already low
        if action == UPGRADE and \
                mappingInfo.metrics.effectiveness < THRESHOLD_EFFECTIVENESS:
            return True

        if temparature <= tempMin:
            return True

        # This is not useful for small number of gpu counts. May be better to use constant
        # threshold value.
        # if iterWithoutSol >= math.pow(3, simData.num_gpus):
        #     return True

        return False

    def shuffleModelList(initialModelsInfo, newModelsInfo, newMappingInfo):
        # TODO: We need to shuffle newModelList so that,
        # sum(abs(initialModelList[idx] - newModelList[idx])) is minimum for all idx
        # Look for a simple optimization function.

        initialModels_sorted = getOrderedModelIDs(initialModelsInfo)
        newModels_sorted = getOrderedModelIDs(newModelsInfo)
        shuffledModelsInfo = {}
        shuffledClientsMap = {}
        shuffledModelsMap = {}

        for i in range(len(initialModels_sorted)):
            initialModel_id = initialModels_sorted[i]
            newModel_id = newModels_sorted[i]
            initialModel = initialModelsInfo[initialModel_id]
            newModel = newModelsInfo[newModel_id]
            client_list = newMappingInfo.models_map[newModel_id]

            newModel.changeGPU(initialModel.gpu_number)
            shuffledModelsInfo[newModel.id] = newModel
            shuffledModelsMap[newModel.id] = client_list

            for client in client_list:
                shuffledClientsMap[client.id] = newModel

        for client_id, model in newMappingInfo.clients_map.items():
            if client_id not in shuffledClientsMap:
                shuffledClientsMap[client_id] = model

        shuffledMappingInfo = MappingInfo(shuffledModelsInfo, newMappingInfo.clients_info,
                                          shuffledModelsMap, shuffledClientsMap, newMappingInfo.name)
        return shuffledMappingInfo

    def _degradeNeighbourGenerator(modelsInfo):
        sorted_models_ids = getOrderedModelIDs(modelsInfo, reverse=True)

        action_lst = [random.randint(-1, 0)
                      for _ in range(len(sorted_models_ids))]
        newModelsInfo = adaptModelsInfo(
            modelsInfo, sorted_models_ids, action_lst)
        return newModelsInfo

    def _uprgadeNeighbourGenerator(modelsInfo):
        sorted_models_ids = getOrderedModelIDs(modelsInfo, reverse=False)

        # Compute likelihood of improvement
        # import numpy as np
        # improvement = np.zeros(len(sorted_models_ids))
        # for i, model_id in enumerate(sorted_models_ids):
        #     modelNumber = modelsInfo[model_id].model_number
        #     nextModelNumber = modelNumber + 1
        #     nextModelNumber = nextModelNumber if nextModelNumber < simData.num_models else (
        #         simData.num_models - 1)
        #     improvement[i] = simData.acc_m[nextModelNumber] -
        #     simData.acc_m[modelNumber]
        #     if improvement[i] < 0.0:
        #         improvement[i] = 0.0
        # prob = improvement / improvement.sum()

        a = np.arange(0, len(sorted_models_ids))
        # idx = np.random.choice(a, p=prob)
        idx = np.random.choice(a)
        model_id = sorted_models_ids[idx]
        # action = random.randint(-1, 1)
        # action = 2
        # newModelsInfo = adaptModels(modelsInfo, model_id, action)

        # some weird probability approach
        # prob = [0.40, 0.30, 0.30]
        # action_lst = [np.random.choice([-1, 0, 1], p=prob)
        #               for _ in range(len(sorted_models_ids))]
        action_lst = [random.randint(-1, 1)
                      for _ in range(len(sorted_models_ids))]
        newModelsInfo = adaptModelsInfo(
            modelsInfo, sorted_models_ids, action_lst)
        return newModelsInfo

    def _getStateID(modelsInfo):
        stateId = ""
        for _, model in sorted(modelsInfo.items(), reverse=False,
                               key=lambda x: x[1].model_number):
            stateId += "_" + str(model.model_number)

        return stateId

    def neighbourGenerator(modelsInfo, action):
        if action == DEGRADE:
            newModelsInfo = _degradeNeighbourGenerator(modelsInfo)
            newStateId = _getStateID(newModelsInfo)
            if newStateId in exploredDegradeStates:
                exploredDegradeStates[newStateId] += 1
            else:
                exploredDegradeStates[newStateId] = 1
        else:
            newModelsInfo = _uprgadeNeighbourGenerator(modelsInfo)
            newStateId = _getStateID(newModelsInfo)
            if newStateId in exploredUpgradeStates:
                exploredUpgradeStates[newStateId] += 1
            else:
                exploredUpgradeStates[newStateId] = 1

        return newModelsInfo

    def reduceTemperature(temparature, iter):
        temparature = temparature * TEMP_REDUCTION_RATIO

        # Geometric schedule
        # temparature = tempInitial * math.pow(TEMP_REDUCTION_RATIO, iter)
        return temparature

    def printStats():
        print('{:=^60}'.format("States Stats"))
        print("\t{:<20s}{:<20s}".format("Upgrade States", "Visited Count"))
        print("\t" + "-"*40)
        for state, count in exploredUpgradeStates.items():
            print("\t{:<20s}{:<20d}".format(state, count))
        print("\t" + "-"*40)
        print("\t{:<20s}{:<20s}".format("Degrade States", "Visited Count"))
        print("\t" + "-"*40)
        for state, count in exploredDegradeStates.items():
            print("\t{:<20s}{:<20d}".format(state, count))

    modelsInfo = initialModelsInfo
    mappingInfo = MappingAlgo(clientsInfo, modelsInfo)

    debugInfo = []
    if debug:
        debugInfo.append(mappingInfo)

    exploredUpgradeStates = {}
    exploredDegradeStates = {}
    mappingCache = Cache()

    # mappingInfo.print()
    # mappingInfo.metrics.print()

    # First see if we need to degrade models to improve the effectiveness metric
    # and then upgrade to improve the accuracy without hurting effectiveness
    for action in [DEGRADE, UPGRADE]:
        bestModelsInfo = modelsInfo
        bestMappingInfo = mappingInfo
        temparature = tempInitial

        iter = 0
        iterWithoutSol = 0
        while not stopCondition(bestMappingInfo, action, temparature, iter, iterWithoutSol):

            newModelsInfo = neighbourGenerator(modelsInfo, action)
            new_state_id = _getStateID(newModelsInfo)

            newMappingInfo = mappingCache.read(new_state_id)
            if newMappingInfo is None:
                newMappingInfo = MappingAlgo(clientsInfo, newModelsInfo)
                mappingCache.update(new_state_id, newMappingInfo)

            if (betterMapping(newMappingInfo, bestMappingInfo, action)):
                if action == DEGRADE:
                    logging.info(
                        "Found better solution: effectiveness before {:.2f} and after {:.2f}".format(bestMappingInfo.metrics.effectiveness,
                                                                                                     newMappingInfo.metrics.effectiveness
                                                                                                     ))
                else:
                    logging.info("Found better solution: accuracy before {:.2f} and after {:.2f}".format(bestMappingInfo.metrics.accuracy_per_request,
                                                                                                         newMappingInfo.metrics.accuracy_per_request,
                                                                                                         ))
                bestModelsInfo = newModelsInfo
                bestMappingInfo = newMappingInfo

                if debug:
                    debugInfo.append(bestMappingInfo)

                # TODO: Should we break?. Or we should explore all neighbours first
                # and then select the best neighbour to build on. Hill climbing explores all neighbours first
                # and selects the best out of all neighbours.
                # break

            if acceptNeighbour(newMappingInfo, mappingInfo, action, temparature):
                modelsInfo = newModelsInfo
                mappingInfo = newMappingInfo
                iterWithoutSol = 0
            else:
                iterWithoutSol += 1

            temparature = reduceTemperature(temparature, iter)
            iter += 1
        modelsInfo = bestModelsInfo
        mappingInfo = bestMappingInfo

        print("Total iter: ", action, iter)
        # if debug:
        #     debugInfo.append(bestMappingInfo)

    # Again reshuffle the model list. This is because to increase the chances of
    # having models already loaded on the GPUs
    if shuffleModels:
        mappingInfo = shuffleModelList(
            initialModelsInfo, modelsInfo, mappingInfo)

    # mappingInfo.print()
    # mappingInfo.metrics.print()
    # printStats()
    return mappingInfo, debugInfo
