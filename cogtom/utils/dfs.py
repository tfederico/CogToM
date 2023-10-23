class Graph:

    def __init__(self, row, col, graph):
        self.ROW = row
        self.COL = col
        self.graph = graph

    # A utility function to do DFS for a 2D
    # boolean matrix. It only considers
    # the 4 neighbours as adjacent vertices
    def DFS(self, i, j):
        if i < 0 or i >= len(self.graph) or j < 0 or j >= len(self.graph[0]) or self.graph[i][j] != 1:
            return None

        # mark it as visited
        self.graph[i][j] = -1

        # Recur for 4 neighbours
        l = self.DFS(i - 1, j)
        u = self.DFS(i, j - 1)
        r = self.DFS(i, j + 1)
        d = self.DFS(i + 1, j)

        pieces = [(i, j)]
        if l is not None:
            pieces += l
        if u is not None:
            pieces += u
        if r is not None:
            pieces += r
        if d is not None:
            pieces += d

        return pieces


    # The main function that returns
    # count of islands in a given boolean
    # 2D matrix
    def countIslands(self):
        # Initialize count as 0 and traverse
        # through the all cells of
        # given matrix
        count = 0
        islands = []
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet,
                # then new island found
                if self.graph[i][j] == 1:
                    # Visit all cells in this island
                    # and increment island count
                    islands.append(self.DFS(i, j))
                    count += 1

        return count, islands
