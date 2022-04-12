module elementwise_array#(parameter num = 4)
(
	// interface to system
    input wire clk,
    input wire reset,
    input wire en,
    input wire [3:0] func,                        // select which func to perform
    // interface to PE row .....

    input wire signed [31:0] in [num-1:0],
    output reg signed [31:0] out [num-1:0]        

	);


    genvar gi;
    generate
        for (gi = 0; gi < num; gi = gi + 1)
        begin:label1
            elementwise_unit elementwise_unit(
                .clk(clk),
                .reset(reset),
                .en(en),
                .func(func),
                .in(in[gi]),
                .out(out[gi])
            );
        end
    endgenerate

endmodule