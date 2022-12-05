%%time
def Persistent_Homology(data,epsilon,k):
  graph = buildGraph(raw_data=data, epsilon=epsilon) #epsilon = 9 will build a "maximal complex"
  ripsComplex = ripsFiltration(graph, k=k)
  # drawComplex(origData=data, ripsComplex=ripsComplex[0])
  BoundaryMatrix = filterBoundaryMatrix(ripsComplex)
  Reducced_BoundaryMatrix = reduceBoundaryMatrix(BoundaryMatrix)
  intervals = readIntervals(Reducced_BoundaryMatrix, ripsComplex[1])
  persist = readPersistence(intervals, ripsComplex)
  graph_barcode(persist, 0)
  graph_barcode(persist, 1)
