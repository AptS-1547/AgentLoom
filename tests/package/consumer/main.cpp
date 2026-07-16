#include <AgentLoom/core/memory_pool.h>

int main() {
    core::BucketMemoryPool pool;
    auto block = pool.allocate(256);
    return block.ok() && block.value().size() == 256 ? 0 : 1;
}
