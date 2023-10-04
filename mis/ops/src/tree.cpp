#include <queue>


typedef std::vector<std::pair<int, int> > Tree;


std::vector<int> get_leaf_descendents(const Tree &tree, int root) {
    int i, n_leaves = tree.size() + 1;

    std::vector<int> descendents;
    std::queue<int> nodes;
    nodes.push(root);

    if (root < n_leaves) {
        descendents.push_back(root);
        return descendents;
    }

    while (!nodes.empty()) {
        i = nodes.front();
        nodes.pop();
        if (i < n_leaves)
            descendents.push_back(i);
        else {
            nodes.push(tree[i - n_leaves].first);
            nodes.push(tree[i - n_leaves].second);
        }
    }

    return descendents;
}


IntMatrix get_roots(const Tree &tree) {
    int i, j, n_leaves = tree.size() + 1;
    IntMatrix roots(n_leaves, n_leaves);
    for (i = 0; i < n_leaves; i++)
        roots(0, i) = i;

    for (i = 1; i < n_leaves; i++) {
        for (j = 0; j < n_leaves; j++) {
            if (roots(i - 1, j) == tree[i - 1].first || roots(i - 1, j) == tree[i - 1].second)
                roots(i, j) = i - 1 + n_leaves;
            else
                roots(i, j) = roots(i - 1, j);
        }
    }

    return roots;
}
