import numpy as np

def find_elbow(varience):
    """
    Find elbow point in scree plot data.
    
    Parameters:
        varience : array - Varience explained values
    
    Returns:
        elbow_x : component at elbow
        elbow_index : index of elbow
        relevant_components : components to keep
    """

    x = np.linspace(1,len(varience),len(varience))
    y = np.array(varience)

    # Normalize data (important for generic datasets)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Line between first and last points
    line_vec = np.array([x_norm[-1] - x_norm[0],
                         y_norm[-1] - y_norm[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)

    # Compute distance from each point to the line
    distances = []
    for i in range(len(x_norm)):
        point_vec = np.array([x_norm[i] - x_norm[0],
                              y_norm[i] - y_norm[0]])

        proj_len = np.dot(point_vec, line_vec)
        proj_vec = proj_len * line_vec

        dist_vec = point_vec - proj_vec
        distances.append(np.linalg.norm(dist_vec))

    elbow_index = np.argmax(distances)
    elbow_x = x[elbow_index]

    relevant_components = x[:elbow_index + 1]

    return elbow_x, elbow_index, relevant_components
