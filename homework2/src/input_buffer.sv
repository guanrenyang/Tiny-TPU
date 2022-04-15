// input buffer, ref: shifter_buffer
//10kb
module input_buffer#(parameter depth = 8)
(
    output reg  [127:0] q,
    input  wire         clk,
    input  wire         reset,
    input  wire         ren,
    input  wire         wen,
    input  wire [5:0]  a,
    input  wire [127:0] d
    );



    integer i;
    integer j;
    reg [127:0] mem [depth-1:0];
    always @(posedge clk) begin
        if(~reset)begin
            for (i = 0; i < depth; i = i + 1)
                mem[i] <= 128'b0;
        end
        else if(wen & ~ren) begin
            q <= 128'd0;
            mem[a] <= d;
        end else if(ren & ~wen) begin
            q <= mem[a];
        end else begin
            q <= 128'd0;
        end
    end

endmodule