import numpy as np
import gudhi
import scipy
import scipy.io
import os
from scipy.optimize import minimize
from itertools import chain, combinations


def f(x):
    # Shannon entropy
    return (x * (np.log(x + (x < 0) * (-x) + 1e-12) - 1)).sum(axis=1)


def f_grad(x):
    # Shannon entropy gradient
    x = x + (x < 0) * (-x)
    return np.log(x + 1e-12)


def KL(x, y):
    # D_f for Shannon entropy (Kullback-Leibler divergence)
    x = x + (x < 0) * (-x)
    y = y + (y < 0) * (-y)

    return (x * np.log(x / (y + 1e-12) + 1e-12) - x + y).sum(axis=1)


def CircumBall(Q, f, f_grad, method='BFGS', verbose=False):
    # Q = 2-dim array of points in simplex
    # f = callable function of Legendre type
    # f_grad = callable function that return gradient of f()
    
    # return: (center, radius)
    
    if Q.shape[0] == 1:
        return Q[0], 0
    f_Q = f(Q)
    hull = np.concatenate((Q, f_Q[:, None]), axis=1)    
    x0 = np.full(hull.shape[0] - 1, 1 / hull.shape[0])
    
    F_opt = lambda x: f((hull[1:].T @ x + (1 - x.sum()) * hull[0])[None, 0:12]) - (hull[1:, 12].T @ x + (1 - x.sum()) * hull[0, 12])

    if verbose:
        print('F_opt in x0:', F_opt(x0))
    F_opt_grad = lambda x: ((hull[1:, 0:12] - hull[0, 0:12]) @ f_grad((hull[1:].T @ x + (1 - x.sum()) * 
                                                                      hull[0])[None, 0:12]).T - (hull[1:, 12] - 
                                                                                                 hull[0, 12])[:, None]).flatten()
    
    
    z = minimize(fun=F_opt, x0=x0, method=method, jac=F_opt_grad)
    
    if verbose:
        print(z)
    
    q = (hull[1:].T @ z.x + (1 - z.x.sum()) * hull[0])[0:12]
    r = -z.fun

    return q, r


def CechRadius(arr, k):
    # arr = 2-dim array of points
    # k = max simplex size (k-skeleton)
    
    # return: {simplex: (center, radius)}
    
    n = len(arr)

    marked = {}
    for i in range(1, k + 1):
        for P in list(combinations(range(n), i)):
            if P not in marked.keys():
                q, r = CircumBall(arr[P, :], f, f_grad, 'BFGS')
                marked[P] = (q, r)
            for a in range(n):
                if a not in P:
                    if KL(arr[a: a+1], q) < r:
                        marked[tuple(sorted(list(P) + [a]))] = (q, r)
    
    return marked


def BuildFiltration(marked, compute_persistence=True):
    # marked = dict {simplex: (center of circumball, radius)}
    # compute_persistence = bool: compute persistence for built filtration
    
    # return: gudhi.SimplexTree() of corresponding filtration
    st = gudhi.SimplexTree()
    for simplex, val in marked.items():
        st.insert(sorted(list(simplex)), filtration=val[1])
    
    if compute_persistence:
        st.compute_persistence()
    return st


def GetStatistics(pers_hom):
    # pers_hom = array of starts and ends of homologies
    
    # return: [mean, std, entropy] of given array
        
    if len(pers_hom) == 0:
        return [0, 0, 0]
    
    ph = pers_hom[np.where(np.isfinite(pers_hom[:, 1]))]
    l = ph[:, 1] - ph[:, 0]
    
    if len(l) == 0:
        mean = 0
        std = 0
    else:
        mean = np.mean(l)
        std = np.std(l)
    
    m = np.max(ph[:, 1])
    ph_ = np.nan_to_num(pers_hom, posinf=m+1)
    
    l_ = ph_[:, 1] - ph_[:, 0]
    p = l_ / l_.sum()
    
    entropy = -(np.log(p + 1e-12) * p).sum()
    
    return [mean, std, entropy]


def ProcessMatrices(mat_folder, max_homology_dim=2, normalize=True, columns_as_vertices=False):
    # mat_folder = folder with pitch transition matrices
    # max_homology_dim = max dimension of homologies
    
    # return: statistics of persistent homologies of matrices
    
    
    mat = []
    for i in range(4):
        mat.append(scipy.io.loadmat(os.path.join(mat_folder, f"{i+1}.mat")))
    
    arr = []
    for i in range(4):
        matrix = np.array(mat[i][f'nmpitch{i+1}'])
        if columns_as_vertices:
            matrix = matrix.T
        arr.append(matrix)
        if normalize:
            arr[i] = (arr[i] / (arr[i].sum(axis=1) + 1e-6)[:, None])
    
    stats = []
    phs = []
    for i in range(4):
        marked = CechRadius(arr[i], max_homology_dim + 1)
        st = BuildFiltration(marked)
        for j in range(max_homology_dim):
            ph = st.persistence_intervals_in_dimension(j)
            stats += GetStatistics(ph)
            phs.append(ph)
    
    return stats, phs


def ParseHTML(corrupted_files):
    composers = ['Beethoven', 'Haydn', 'Mozart']
    genres = dict(zip(composers, [{}, {}, {}]))
    names_for_genres = dict(zip(composers, [{}, {}, {}]))

    with open("html/Be.htm", "r") as f:
        file = f.read()
        search='href="MIDIfiles/Beethoven/BeethovenSQ/'
        while True:
            file = file[file.find(search) + len(search):]
            if file[0:9] != 'Beethoven':
                break
            name, genre = file.split('/')[1].split('\"')[0:2]
            name = name[:-3] + 'mid'
            if name in corrupted_files:
                continue
            genre=genre.split('&')[0].replace('<', '').replace('>', '').replace('\n', '').strip()
            genres['Beethoven'][name] = genre
            names_for_genres['Beethoven'][genre] = names_for_genres['Beethoven'].get(genre, set()).union(set([name]))


    with open("html/Ha.htm", "r") as f:
        file = f.read()
        search='href="MIDIfiles/Haydn/HaydnSQ/'
        while True:
            file = file[file.find(search) + len(search):]
            if file[0:5] != 'Haydn':
                break
            name, genre = file.split('/')[1].split('\"')[0:2]
            name = name[:-3] + 'mid'
            if name in corrupted_files:
                continue
            genre=genre.split('&')[0].replace('<', '').replace('>', '').replace('\n', '').strip()
            genres['Haydn'][name] = genre
            names_for_genres['Haydn'][genre] = names_for_genres['Haydn'].get(genre, set()).union(set([name]))


    with open("html/Mo.htm", "r") as f:
        file = f.read()
        search='href="MIDIfiles/Mozart/MozartSQ/'
        while True:
            file = file[file.find(search) + len(search):]
            if file[0:6] != 'Mozart':
                break
            name, genre = file.split('/')[1].split('\"')[0:2]
            name = name[:-3] + 'mid'
            if name in corrupted_files:
                continue
            genre=genre.split('&')[0].replace('<', '').replace('>', '').replace('\n', '').strip()
            genres['Mozart'][name] = genre
            names_for_genres['Mozart'][genre] = names_for_genres['Mozart'].get(genre, set()).union(set([name]))
        
    return genres, names_for_genres