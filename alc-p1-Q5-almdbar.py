import numpy as np


def resolve_lu(A: np.ndarray, b: np.ndarray, mostrarPassos: bool = True):
    U = np.array(A, dtype=float)
    bVec = np.array(b, dtype=float)
    n = len(U)
    L = np.eye(n)
    if mostrarPassos:
        stepMatrix(U, "A")

    for pivot in range(n):
        if U[pivot][pivot] == 0:
            raise Exception(
                f"Pivô nulo em U[{pivot}][{pivot}]: a decomposição LU sem pivoteamento não pode continuar. "
                "Utilize uma função alternativa para resolver o sistema "
            )
        if mostrarPassos:
            print(f"Pivô atual: U[{pivot}][{pivot}] = {U[pivot][pivot]}")

        for lineBelow in range(pivot + 1, n):
            mult = U[lineBelow][pivot] / U[pivot][pivot]
            L[lineBelow][pivot] = mult
            if mostrarPassos:
                print(f"  m[{lineBelow}][{pivot}] = {U[lineBelow][pivot]} / {U[pivot][pivot]} = {mult}"
                      f"  ->  L[{lineBelow}][{pivot}] = {mult}"
                      f"  |  linha {lineBelow} <- linha {lineBelow} - ({mult}) * linha {pivot}")
            U[lineBelow][pivot] = 0.0
            for col in range(pivot + 1, n):
                U[lineBelow][col] -= mult * U[pivot][col]

        if mostrarPassos and pivot < n - 1:
            stepMatrix(U, f"U após eliminar a coluna {pivot}")
            stepMatrix(L, "L até aqui")

    y = substProgressiva(L, bVec)
    x = substRegressiva(U, y)
    if mostrarPassos:
        stepMatrix(L, "L")
        stepMatrix(U, "U")
        print(f"y (Ly = b) = {y}")
        print(f"x (Ux = y) = {x}")
    return L, U, x


def stepMatrix(M: np.ndarray, titulo: str = ""):
    if titulo:
        print(f"{titulo} =")
    for linha in M:
        print("| " + " ".join(f"{valor:10.4f}" for valor in linha) + " |")
    print()


def substProgressiva(L: np.ndarray, b: np.ndarray):
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = (b[i] - soma) / L[i][i]
    return y


def substRegressiva(U: np.ndarray, y: np.ndarray):
    n = len(U)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]
    return x


if __name__ == "__main__":
    A = np.array([[4, 3, 2],
                  [2, 1, 3],
                  [3, 2, 1]])
    b = np.array([1, 2, 3])
    L, U, x = resolve_lu(A, b)