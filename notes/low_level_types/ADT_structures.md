## Roaring Bitmaps

https://arxiv.org/pdf/1402.6407v4 - Better bitmap performance with Roaring bitmaps

Notes:

- Roaring bitmaps partition range of 32 bit indices into chunks of 2^16 ints sharing the same 16 most significant digits (first level)
  - Other containers store the remaining bits (second level)
- Use array for sparse chunks and a bitmap for dense chunks
  - Allows first level index to stay in cache
- Containers store cardinality, so computing only requires summing over the (low) number of containers
  - Supports rank and select
- Density num_of_ints / num_of_containers should be > 0.1%, otherwise don't use bitmap
- Lookups use binary search
- Operations use:
  - iterating over sorted first level arrays - unions go through both operands, intersection terminates after first operand iteration is complete
    - finding cardinality of new bitsets is fast since processors have dedicated instructions to compute the # of 1s in a word
  - separate methods for bitmap vs bitmap, bitmap vs array, array vs array
  - support in-place operations
