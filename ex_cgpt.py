
import numpy as np

class TrackedArray:
    """Classe qui intercepte à la fois les ufuncs et les fonctions numpy."""
    
    def __init__(self, data, name="A"):
        self.data = np.asarray(data)
        self.name = name

    def __repr__(self):
        return f"TrackedArray({self.name}, data={self.data})"

    # --- Interception des opérations ufuncs ---
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(f"[__array_ufunc__] {ufunc.__name__} appelé sur {[getattr(i, 'name', i) for i in inputs]}")
        
        # Dérive les données sous-jacentes
        unwrapped = [i.data if isinstance(i, TrackedArray) else i for i in inputs]
        result = getattr(ufunc, method)(*unwrapped, **kwargs)
        
        # Renvoie un TrackedArray si le résultat est un ndarray
        if isinstance(result, np.ndarray):
            return TrackedArray(result, name=f"({'+'.join(getattr(i, 'name', str(i)) for i in inputs)})")
        else:
            return result

    # --- Interception des fonctions numpy de haut niveau ---
    def __array_function__(self, func, types, args, kwargs):
        print(f"[__array_function__] {func.__name__} appelé avec {[getattr(a, 'name', a) for a in args]}")
        
        # Applique la fonction standard sur les données réelles
        unwrapped_args = [a.data if isinstance(a, TrackedArray) else a for a in args]
        result = func(*unwrapped_args, **kwargs)
        
        # Si le résultat est un ndarray, on le ré-emballe
        if isinstance(result, np.ndarray):
            return TrackedArray(result, name=f"{func.__name__}_result")
        else:
            return result


# === Démonstration ===
if __name__ == "__main__":
    a = TrackedArray([1, 2, 3], name="A")
    b = TrackedArray([10, 20, 30], name="B")

    print("\n--- Étape 1 : multiplication élémentaire (ufunc) ---")
    c = a * b   # np.multiply(a, b)
    print("Résultat:", c)

    print("\n--- Étape 2 : moyenne du résultat (fonction numpy) ---")
    m = np.mean(c)
    print("Résultat final:", m)
