module PE(
	// interface to system
    input wire clk,
    input wire reset,
    input wire c_en,                              // compute enable
    input wire p_en,                              // preload enable
    // interface to PE row .....

    input wire signed[31:0]in_weight,             // wire from weight buffer direction

    input wire signed[31:0]in_input,              // wire from input buffer direction
     
    output wire signed[31:0]result
	);


reg signed [31:0] accu_sum_reg;





endmodule