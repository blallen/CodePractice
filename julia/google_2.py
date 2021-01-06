"""
You dug up a treasure with a lock that has N binary dials. The lock can be unlocked by turning one dial at a time towards the unlock pattern. 
WARNING: only some patterns are safe (you can access them in a global variable SAFE). Any other pattern will cause the destruction of the contents inside.
Your task is to write a function that will accept a lock description: the current state of dials and the unlock sequence. 
Your function should return "TREASURE" if it's possible to unlock the treasure, and "IMPOSSIBLE" otherwise. As a secondary goal, 
you'd love to do this as quickly as possible, and record the sequence used.

Example 1:
Initial pattern: 010 Unlocked pattern: 111 Safe patterns: 000 001 010 101 111
Correct response is: TREASURE because there is a safe sequence from the initial pattern to the unlock pattern: 010 -> 000 -> 001 -> 101 -> 111
Example 2:
Initial pattern: 00 Unlocked pattern: 11 Safe patterns: 00 11
Correct response is: "IMPOSSIBLE" as no sequence can go from 00 to 11 with one dial turn at a time.
Example 3:
Initial pattern: 00 Unlocked pattern: 11 Safe patterns: 00 01 11
Correct response is: TREASURE because there is a safe sequence to the unlock pattern (00->01->11).
"""

SAFE = []

# sequence = [00, 01, 11]
# sequence = [010, 000, 001, 101, 111]

def findTreasure(start, final, sequence = []):
  if length(final) != length(start):
    return "Impossible"
  
  for i in range(0, length(start)):
    bit = start[i]
    bit = 0 if bit else bit = 1
    
    flip = copy(start)
    flip[i] = bit
    
    if flip is final:
      sequence.append(flip)
      return "TREASURE"
    
    else if flip is in sequence:
      continue
    
    else if flip is in SAFE:
      sequence.append(start)
      return findTreasure(start, final)
  
  return "Impossible"

"""
You own an ice cream shop, and want to help undecided customers with their flavor choices. 
You have a history of all of the flavor combinations that people have bought in the past six months. 
Write an algorithm that suggests a another scoop, given that the customer has picked up to two themselves.
"""

Customers = { Name : FlavorGraph }

def FlavorGraph: 
  flavors
  Adjacency[flavor]
  
  Chocolate->Vanilla->Strawberry
  
# listOfChoices = [(Chocolate, Vnilla), (Straweberry)]
def createFlavorGraph(listOfChoices):
  graph = FlavorGraph()
  
  for choice in listOfChoices:
    for flavor in choice:
      otherFlavors = choice.remove(flavor)
      
      if flavor not in graph.vertices():
        node = Flavor(flavor)
        
        graph.Adjacency(flavor) = otherFlavors
        
      else:
        for otherFlavor in otherFlavors:
          graph.weight[graph.edges((flavor, otherFlavor))] += 1