# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 23:18:01 2021

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:11:49 2021

@author: Mostafa
"""

def Dijkstra(mat,src,dest):
    
    dv={}
    
    finished_nodes=[src]
    
#    print(mat[0])
    
    for i in range(len(mat)):
        if not i==src:
            dv[i]=mat[src][i],src
#            print(mat[src][i])
    
    for _ in range(len(mat)-1):
        
        print(dv)
        
        min_tuple=min(dv.items(),key=lambda x: x[1])
#        print(min_tuple)
        min_distance=min_tuple[1][0]
        min_node=min_tuple[0]
        finished_nodes.append(min_node)
        
        for j in range(1+len(dv)):
#            print(j)
            if j in finished_nodes:
                continue
            
            if dv[j][0]>min_distance+mat[min_node][j]:
                dv[j]=min_distance+mat[min_node][j],min_node
        
    return dv

#while 1:
#    
#    x=input('')
#    
#    if x=='Add Command':
#
x=[
   [0,3,1,1e8],
   [3,0,1,1],
   [1,1,0,3],
   [1e8,1,3,0],
   ]

src=0

o=Dijkstra(x,src,2)
#o.o.items

def Dijkstra1(NodeConnectionDict,Source,isbidirectional=False):
    
    ActiveNodes=set()
    for i,j in NodeConnectionDict:
        ActiveNodes.add(i)
        ActiveNodes.add(j)
#        ActiveNode
    
#    ActiveNodes.x={}
    
    
    for i in ActiveNodes:
    
    return ActiveNodes
    
    
    
    
    
    
if __name__=='__main__':
    
    d={
       (1,2): 100,
       (2,3): 10,
       (3,1): 200
       }
#    x={1,2,3}
#    y={4,5,6}
    
    print(Dijkstra(d,1))