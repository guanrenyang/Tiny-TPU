module PE_array#(parameter num = 4)
(
	// interface to system
    input wire clk,
    input wire reset,
    input wire c_en,                              // compute enable
    input wire p_en,                              // preload enable
    // interface to PE row .....

    input wire signed[31:0]in_weight[num-1:0],             // wire from weight buffer direction

    input wire signed[31:0]in_input[num-1:0], 
                 // wire from input buffer direction
    output signed[31:0]result[num-1:0],

    output compute_finished

	);
    // some hint and suggestions but not compulsory:
    //1. you can use a set of shift resigters for data alignment before computing.
    //2. you can use  combinational circuits as the transpose unit after you
    // get the complete matrix, due to its simplicity.Maybe extra storage is needed.
    //3. you can use shift logic to offload the result from the PE_array.
    //4. only if the PE_array's state is computing ,the data should be aligned in parallelogram,
    //otherwise, it should be aligned in rectangle.
    //5. you need to calculte the compute_finished signal taking the transpose and computing etc into consideration.

    // Spec
    // input: input data belonging to the same row/column of an input matrix from weight/input buffer should arrive at PE array at the same time
    // output: input data belonging to the same row/column of an output matrix should leave PE array at the same time





    // to do: insert buffers in front of different rows or columns of PE array to ensure logic correction






    // to do: some glue jobs






endmodule
