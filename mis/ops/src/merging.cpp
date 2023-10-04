#define DEFAULT_ADJACENT_SIZE 8


typedef std::vector<std::vector<int> > Adjacent;
typedef std::pair<float, std::pair<int, int> > CostItem;


void compute_ward_cost(
    FloatArray &cost,
    const FloatMatrix &centers,
    const FloatArray &sizes,
    const std::vector<int> &indices_src,
    const std::vector<int> &indices_tgt
) {
    FloatMatrix centers_src = centers(indices_src, Eigen::all);
    FloatMatrix centers_tgt = centers(indices_tgt, Eigen::all);
    FloatArray sizes_src = sizes(indices_src);
    FloatArray sizes_tgt = sizes(indices_tgt);
    FloatArray diff = (centers_src - centers_tgt).rowwise().squaredNorm();
    cost = (sizes_src * sizes_tgt) * (sizes_src + sizes_tgt).inverse() * diff;
}


bool compare_cost(const CostItem &x, const CostItem &y) {
    return x.first > y.first;
}


void init_adjacent(
    Adjacent &adjacent,
    std::vector<int> &indices_src,
    std::vector<int> &indices_tgt,
    const int height,
    const int width
) {
    int k = -1;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i > 0)
                adjacent[i * width + j].push_back((i - 1) * width + j);
            if (j > 0)
                adjacent[i * width + j].push_back(i * width + j - 1);
            if (i < height - 1) {
                adjacent[i * width + j].push_back((i + 1) * width + j);
                indices_src[++k] = i * width + j;
                indices_tgt[k] = (i + 1) * width + j;
            }
            if (j < width - 1) {
                adjacent[i * width + j].push_back(i * width + j + 1);
                indices_src[++k] = i * width + j;
                indices_tgt[k] = i * width + j + 1;
            }
        }
    }
}


void update_adjacent(
    const int tgt,
    const int src,
    Adjacent &adjacent,
    std::vector<int> &parent,
    std::vector<bool> &is_visited
) {
    int n, p;
    for (auto it = adjacent[src].begin(); it != adjacent[src].end(); it++) {
        n = *it;
        p = parent[n];
        while (p != n) {
            n = p;
            p = parent[n];
            parent[n] = parent[p];
        }
        if (!is_visited[n]) {
            is_visited[n] = true;
            adjacent[tgt].push_back(n);
        }
    }
}


Tree bottom_up_merging(Eigen::Ref<FloatMatrix> features, int height, int width) {
    if (height * width != features.rows())
        throw std::invalid_argument("The number of pixels must be equal to the number of rows of features.");

    int n_leaves = features.rows();
    int n_features = features.cols();
    int n_nodes = n_leaves * 2 - 1;
    int i, j, k;

    Tree tree;
    tree.reserve(n_leaves - 1);

    FloatMatrix centers(n_nodes, n_features);
    FloatArray sizes = FloatArray::Zero(n_nodes);
    centers(Eigen::seq(0, n_leaves - 1), Eigen::all) = features;
    sizes(Eigen::seq(0, n_leaves - 1)) = FloatArray::Ones(n_leaves);

    std::vector<int> _init_neighbors;
    _init_neighbors.reserve(DEFAULT_ADJACENT_SIZE);
    Adjacent adjacent(n_nodes, _init_neighbors);
    int n_items = (height - 1) * (width - 1) * 2 + height + width - 2;
    std::vector<int> indices_src(n_items);
    std::vector<int> indices_tgt(n_items);
    init_adjacent(adjacent, indices_src, indices_tgt, height, width);

    FloatArray cost;
    compute_ward_cost(cost, centers, sizes, indices_src, indices_tgt);

    std::vector<CostItem> cost_heap(cost.size());
    for (i = 0; i < cost.size(); i++)
        cost_heap[i] = (std::make_pair(cost[i], std::make_pair(indices_src[i], indices_tgt[i])));
    std::make_heap(cost_heap.begin(), cost_heap.end(), compare_cost);

    std::vector<bool> is_available(n_nodes, true);
    std::vector<int> parent(n_nodes);
    for (i = 0; i < n_nodes; i++)
        parent[i] = i;

    for (k = n_leaves; k < n_nodes; k++) {
        while (1) {
            std::pop_heap(cost_heap.begin(), cost_heap.end(), compare_cost);
            CostItem cost_item = cost_heap.back();
            cost_heap.pop_back();
            i = cost_item.second.first;
            j = cost_item.second.second;
            if (is_available[i] && is_available[j])
                break;
        }

        parent[i] = parent[j] = k;
        tree.push_back(std::make_pair(i, j));
        is_available[i] = is_available[j] = false;

        centers(k, Eigen::all) = centers(i, Eigen::all) * sizes(i) + centers(j, Eigen::all) * sizes(j);
        sizes(k) = sizes(i) + sizes(j);
        centers(k, Eigen::all) /= sizes(k);

        std::vector<bool> is_visited(n_nodes, false);
        is_visited[k] = true;

        update_adjacent(k, i, adjacent, parent, is_visited);
        update_adjacent(k, j, adjacent, parent, is_visited);

        compute_ward_cost(cost, centers, sizes, std::vector<int>(adjacent[k].size(), k), adjacent[k]);

        for (i = 0; i < adjacent[k].size(); i++) {
            cost_heap.push_back(std::make_pair(cost[i], std::make_pair(adjacent[k][i], k)));
            std::push_heap(cost_heap.begin(), cost_heap.end(), compare_cost);
        }
    }

    return tree;
}
