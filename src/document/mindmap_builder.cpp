#include "document_types.h"
#include "semantic_cache_pipeline.h"

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>

namespace agent::document {
namespace {

struct MindmapNode {
    std::string name;
    int level = 0;
    std::string block_id;
    std::optional<std::uint64_t> order;
    std::optional<int> page;
    std::optional<int> slide;
    std::string text;
    std::vector<std::unique_ptr<MindmapNode>> children;
};

nlohmann::json ToJson(const MindmapNode& node) {
    nlohmann::json children = nlohmann::json::array();
    for (const auto& child : node.children) {
        children.push_back(ToJson(*child));
    }
    return {
        {"name", node.name},
        {"children", children},
    };
}

std::string RootName(std::string_view file_name) {
    auto name = std::string(file_name);
    const auto slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    const auto dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name = name.substr(0, dot);
    }
    return name.empty() ? "Teaching Material" : name;
}

std::string ShortLeaf(std::string_view text, std::size_t limit = 48) {
    auto clean = CleanText(text);
    if (clean.size() <= limit) {
        return clean;
    }
    std::size_t end = 0;
    const auto max_bytes = limit > 3 ? limit - 3 : limit;
    for (std::size_t i = 0; i < clean.size() && i < max_bytes;) {
        const auto ch = static_cast<unsigned char>(clean[i]);
        std::size_t len = 0;
        if (ch < 0x80) {
            len = 1;
        } else if ((ch & 0xE0) == 0xC0) {
            len = 2;
        } else if ((ch & 0xF0) == 0xE0) {
            len = 3;
        } else if ((ch & 0xF8) == 0xF0) {
            len = 4;
        } else {
            break;
        }
        if (i + len > clean.size() || i + len > max_bytes) {
            break;
        }
        bool valid = true;
        for (std::size_t j = 1; j < len; ++j) {
            if ((static_cast<unsigned char>(clean[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            break;
        }
        i += len;
        end = i;
    }
    return clean.substr(0, end) + "...";
}

using EmbeddingVector = std::vector<float>;

bool IsSeparatorText(std::string_view text) {
    const auto clean = CleanText(text);
    if (clean.empty()) {
        return true;
    }
    return clean.find_first_not_of("-_=*#—") == std::string::npos;
}

double PositionScore(const ChunkTrunk& chunk, const MindmapNode& heading) {
    if (!heading.order) {
        return 0.1;
    }
    const auto distance = chunk.order > *heading.order
        ? chunk.order - *heading.order
        : *heading.order - chunk.order;
    return 1.0 / (1.0 + static_cast<double>(distance));
}

std::string ChunkEmbeddingText(const ChunkTrunk& chunk) {
    return CleanText(chunk.title + " " + chunk.summary + " " + chunk.text.substr(0, 1200));
}

std::string HeadingEmbeddingText(const MindmapNode& heading) {
    return CleanText(heading.text.empty() ? heading.name : heading.text);
}

std::optional<EmbeddingVector> EmbedText(std::string_view text,
                                         const std::shared_ptr<IDocumentEmbeddingProvider>& provider) {
    if (!provider || text.empty()) {
        return std::nullopt;
    }
    auto embedded = provider->EmbedText(text);
    if (!embedded.ok() || embedded.value().size() != agent::semantic_cache::kExpectedEmbeddingDim) {
        return std::nullopt;
    }
    return std::move(embedded).value();
}

float EmbeddingSimilarity(const EmbeddingVector& lhs, const EmbeddingVector& rhs) {
    if (lhs.size() != agent::semantic_cache::kExpectedEmbeddingDim ||
        rhs.size() != agent::semantic_cache::kExpectedEmbeddingDim) {
        return 0.0f;
    }
    return dot_product_unrolled<agent::semantic_cache::kExpectedEmbeddingDim>(lhs.data(), rhs.data());
}

double ScopeScore(const ChunkTrunk& chunk, const MindmapNode& heading) {
    if (chunk.slide && heading.slide == chunk.slide) {
        return 1.0;
    }
    if (chunk.page && heading.page == chunk.page) {
        return 0.8;
    }
    return 0.0;
}

std::vector<MindmapNode*> CandidateHeadings(const ChunkTrunk& chunk, const std::vector<MindmapNode*>& headings) {
    std::vector<MindmapNode*> scoped;
    for (auto* heading : headings) {
        if ((chunk.slide && heading->slide == chunk.slide) || (chunk.page && heading->page == chunk.page)) {
            scoped.push_back(heading);
        }
    }
    if (!scoped.empty()) {
        std::sort(scoped.begin(), scoped.end(), [&](const MindmapNode* left, const MindmapNode* right) {
            const auto lo = left->order.value_or(0);
            const auto ro = right->order.value_or(0);
            const auto ld = chunk.order > lo ? chunk.order - lo : lo - chunk.order;
            const auto rd = chunk.order > ro ? chunk.order - ro : ro - chunk.order;
            return ld < rd;
        });
        if (scoped.size() > 6) {
            scoped.resize(6);
        }
        return scoped;
    }

    std::vector<MindmapNode*> previous;
    std::vector<MindmapNode*> following;
    for (auto* heading : headings) {
        if (!heading->order) {
            continue;
        }
        if (*heading->order <= chunk.order) {
            previous.push_back(heading);
        } else {
            following.push_back(heading);
        }
    }
    std::vector<MindmapNode*> candidates;
    const auto prev_start = previous.size() > 4 ? previous.size() - 4 : 0;
    candidates.insert(candidates.end(), previous.begin() + static_cast<std::ptrdiff_t>(prev_start), previous.end());
    candidates.insert(candidates.end(), following.begin(), following.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(2, following.size())));
    if (!candidates.empty()) {
        return candidates;
    }
    const auto start = headings.size() > 6 ? headings.size() - 6 : 0;
    return std::vector<MindmapNode*>(headings.begin() + static_cast<std::ptrdiff_t>(start), headings.end());
}

MindmapNode* SelectParentForChunk(const ChunkTrunk& chunk,
                                  const std::vector<MindmapNode*>& headings,
                                  MindmapNode& root,
                                  const std::shared_ptr<IDocumentEmbeddingProvider>& embedding_provider,
                                  std::unordered_map<const MindmapNode*, std::optional<EmbeddingVector>>& heading_embeddings) {
    if (headings.empty()) {
        return &root;
    }
    auto candidates = CandidateHeadings(chunk, headings);
    auto* best = candidates.front();
    double best_score = -1.0;
    auto chunk_embedding = EmbedText(ChunkEmbeddingText(chunk), embedding_provider);
    for (auto* heading : candidates) {
        double semantic = 0.0;
        if (chunk_embedding) {
            auto it = heading_embeddings.find(heading);
            if (it == heading_embeddings.end()) {
                it = heading_embeddings.emplace(heading, EmbedText(HeadingEmbeddingText(*heading), embedding_provider)).first;
            }
            if (it->second) {
                semantic = static_cast<double>(EmbeddingSimilarity(*chunk_embedding, *it->second));
            }
        }
        const auto score = 0.55 * semantic + 0.35 * PositionScore(chunk, *heading) + 0.10 * ScopeScore(chunk, *heading);
        if (score > best_score) {
            best_score = score;
            best = heading;
        }
    }
    return best;
}

} // namespace

nlohmann::json BuildMindmap(const std::vector<DocumentBlock>& blocks,
                            std::string_view file_name,
                            const std::vector<ChunkTrunk>& chunks,
                            std::shared_ptr<IDocumentEmbeddingProvider> embedding_provider) {
    MindmapNode root;
    root.name = RootName(file_name);

    if (blocks.empty()) {
        root.children.push_back(std::make_unique<MindmapNode>(MindmapNode{.name = "No extractable text found", .level = 1}));
        return ToJson(root);
    }

    std::set<std::string> chunk_block_ids;
    for (const auto& chunk : chunks) {
        chunk_block_ids.insert(chunk.block_ids.begin(), chunk.block_ids.end());
    }

    std::vector<MindmapNode*> stack{&root};
    std::vector<MindmapNode*> headings;
    MindmapNode* unclassified = nullptr;

    for (const auto& block : blocks) {
        const auto text = CleanText(block.text);
        if (IsSeparatorText(text)) {
            continue;
        }

        auto level = InferHeadingLevel(block);
        if (block.kind == "heading" && level) {
            const auto clamped = std::max(1, std::min(6, *level));
            while (!stack.empty() && stack.back()->level >= clamped) {
                stack.pop_back();
            }
            if (stack.empty()) {
                stack.push_back(&root);
            }
            auto child = std::make_unique<MindmapNode>();
            auto& node = *child;
            node.name = ShortLeaf(text, 64);
            node.level = clamped;
            node.block_id = block.id;
            node.order = block.order;
            node.page = block.page;
            node.slide = block.slide;
            node.text = text;
            stack.back()->children.push_back(std::move(child));
            stack.push_back(&node);
            headings.push_back(&node);
            continue;
        }

        if (chunk_block_ids.contains(block.id)) {
            continue;
        }

        MindmapNode* parent = stack.size() > 1 ? stack.back() : nullptr;
        if (!parent) {
            if (!unclassified) {
                auto child = std::make_unique<MindmapNode>();
                auto& node = *child;
                node.name = "Unclassified Content";
                node.level = 1;
                root.children.push_back(std::move(child));
                unclassified = &node;
            }
            parent = unclassified;
        }
        auto child = std::make_unique<MindmapNode>();
        auto& leaf = *child;
        leaf.name = ShortLeaf(text);
        leaf.level = parent->level + 1;
        leaf.block_id = block.id;
        leaf.order = block.order;
        leaf.page = block.page;
        leaf.slide = block.slide;
        leaf.text = text;
        parent->children.push_back(std::move(child));
    }

    std::unordered_map<const MindmapNode*, std::optional<EmbeddingVector>> heading_embeddings;
    for (const auto& chunk : chunks) {
        auto* parent = SelectParentForChunk(chunk, headings, root, embedding_provider, heading_embeddings);
        auto chunk_child = std::make_unique<MindmapNode>();
        auto& chunk_node = *chunk_child;
        chunk_node.name = ShortLeaf(!chunk.title.empty() ? chunk.title : (!chunk.summary.empty() ? chunk.summary : "Unstructured Content"), 64);
        chunk_node.level = parent->level + 1;
        chunk_node.block_id = chunk.chunk_id;
        chunk_node.order = chunk.order;
        chunk_node.page = chunk.page;
        chunk_node.slide = chunk.slide;
        chunk_node.text = CleanText(chunk.title + " " + chunk.summary);

        for (std::size_t i = 0; i < chunk.slices.size(); ++i) {
            const auto& item = chunk.slices[i];
            const auto label = !item.title.empty() ? item.title : (!item.summary.empty() ? item.summary : (!item.text.empty() ? item.text : "Chunk Item " + std::to_string(i + 1)));
            if (IsSeparatorText(label)) {
                continue;
            }
            auto slice_child = std::make_unique<MindmapNode>();
            auto& child = *slice_child;
            child.name = ShortLeaf(label);
            child.level = chunk_node.level + 1;
            child.block_id = chunk.chunk_id + "-" + std::to_string(i);
            child.order = chunk.order;
            child.page = chunk.page;
            child.slide = chunk.slide;
            child.text = !item.text.empty() ? item.text : (!item.summary.empty() ? item.summary : label);
            chunk_node.children.push_back(std::move(slice_child));
        }
        parent->children.push_back(std::move(chunk_child));
    }

    if (root.children.empty()) {
        root.children.push_back(std::make_unique<MindmapNode>(MindmapNode{.name = "No extractable text found", .level = 1}));
    }
    return ToJson(root);
}

} // namespace agent::document
