#define BLOCK_SIZE 16
#define index(y, x, the_size) (y*the_size + x)

void __kernel matrix_product(
			    float __global *matrix,
			    float __global *results,
			    int size
			     ) { 
  //left matrix is shared_size x height, right matrix is width x shared_size
  float __local  l_block[BLOCK_SIZE][BLOCK_SIZE], r_block[BLOCK_SIZE][BLOCK_SIZE+1];
  //The l_block matrix is padded because we will access its elements column wise.
  //Padding the r_block matrix is unnecessary, since its elements are only accessed by row
  int i, j, blockR, blockC, r, c;
  blockC = get_group_id(0) * BLOCK_SIZE;
  blockR = get_group_id(1) * BLOCK_SIZE;
  c = get_local_id(0);
  r = get_local_id(1);
  float weight;
  weight = INFINITY;


  for(i = 0; i*BLOCK_SIZE < size; i++) { //try using a variable to store shared_size/16?
    //load subblock into local memory
    /*
    left += BLOCK_SIZE;
    right += size*BLOCK_SIZE;
    l_block[r][c] = 
      matrix[left + r*size + c];

    r_block[r][c] = 
    matrix[right + r*size + c];*/
    l_block[r][c] = 
      *(matrix + (blockR+r)*size + (i * BLOCK_SIZE) + c);
    r_block[r][c] = 
      *(matrix + blockC + (i * size * BLOCK_SIZE) + r*size + c);    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(j = 0; j< BLOCK_SIZE; j++) {
      int temp = l_block[r][j]+r_block[j][c];
      if(temp < weight) {
	weight = temp;
      }
    }
    
    //Make sure that we are done with the matrices stored in local memory before we enter the next iteration
    barrier(CLK_LOCAL_MEM_FENCE); 
  }
  
  results[get_global_id(1)*size + get_global_id(0)] = weight;
  //preds[index(get_global_id(1), get_global_id(0), size)] = pred;
}
