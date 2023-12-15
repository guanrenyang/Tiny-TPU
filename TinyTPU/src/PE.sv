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

reg signed [31:0] curr_sum;
always @(posedge clk) begin
    if(~reset) begin
        accu_sum_reg <= 32'b0;
    end
    else begin
        if(c_en && !p_en) begin
            accu_sum_reg <= in_weight * in_input+accu_sum_reg;  
        end else if (!c_en && p_en) begin
            accu_sum_reg <= in_weight;
        end else begin
        
        end
    
    end
end
assign result = accu_sum_reg;


endmodule