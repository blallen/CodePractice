"""
func(rootnode)

Given a large binary tree where each node has integer values, find identical subtrees. That is, find sets of nodes where each node and all its children are identical.

	          1         
          /   \
        2      2      
      /   \  /   \
      x   3  x   3
         / \      / \
         x  x   x  x


->       2    ,   3
       /   \     /  \
       x   3    x   x 
          / \  
          x  x

Node 3
value 3
left -> Node(Nil)
right -> Node(Nil)
valueAncestors = [Nil, Nil]
"""

1 [2[x x] 3[4[x x] 5[x x]]] 

2.valueAncestors - > x 2 x
3.valueAncestors - > x 4 x 3 x 5 x
1.valueAncestors - > x 2 x 1 x 4 x 3 x 5 x

1,2,3,4 - 1,2,3,4
""" 
x means a node with value Nil 
"""

def Node:
    value
    parent
    left
    right
    depth
    valueAncestors = []
    color/visited
    
    
    
def findIdentical(root)
  subTrees = {}
  valueAncestors = []
  
  findAncestors(root, subTrees)
  
  identicalTrees = []
  
  for (tree, nIdentical) in subTrees.iteritems():
    if nIdentical > 1:
      identicalTrees.append(tree)
      
  return identicalTrees
    
    
def findAncestors(node, subTrees):
  if node.left is not None:
    findIdentical(node.left)
    # if node.left.value is not in node.valueAncestors:
    node.valueAncestors += str(node.left.valueAncestors) + str('/')
      
  node.valueAncestors += str(node.value) + str('/')
    
  if node.right is not None:
    findIdentical(node.right)
    # if node.right.value is not in node.valueAncestors:
    node.valueAncestors += str(node.right.valueAncestors)
  
  if node.valueAncestors is not in subTrees:
    subTrees[node.valueAncestors] = 1
  else:
    subTrees[node.valueAncestors] += 1
    
    