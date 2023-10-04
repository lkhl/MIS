std::vector<int> top_down_sampling(const Tree &tree, float decay, int seed) {
    int n_leaves = tree.size() + 1;
    int node = n_leaves * 2 - 2;
    float p = 1.0;

    std::default_random_engine random_engine(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    while (distribution(random_engine) <= p) {
        if (node < n_leaves)
            break;

        if (distribution(random_engine) <= 0.5)
            node = tree[node - n_leaves].first;
        else
            node = tree[node - n_leaves].second;

        p *= decay;
    }

    return get_leaf_descendents(tree, node);
}
