module elementwise_unit(
	// interface to system
    input wire clk,
    input wire reset,
    input wire en,
    input wire [3:0] func,                        // select which func to perform
    // interface to PE row .....

    input wire signed[31:0]in,
    output reg signed[31:0]out

	);



    always @(negedge reset or posedge clk ) begin
        if(~reset) begin
            out <= 32'b0;
        end
        else begin
            if(en) begin
                case(func)
                    4'b0001:       out <= in ;  // pass
                    4'b0010:       out <= (in[31] == 1'b0) ? in : 0 ;  // ReLU
                    // to be extended
                    default:       out <= 32'b0 ;
                endcase
            end
            else begin
                out <= 32'b0;
            end
        end
    end


endmodule