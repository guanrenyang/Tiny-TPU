// input buffer, ref: shifter_buffer
//10kb
module shared_memory#(parameter depth = 64)
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

    reg [127:0] mem [depth-1:0];
    always @(posedge clk) begin
        if(~reset)begin                         // reset when reset==0
            for (i = 0; i < depth; i = i + 1)
                mem[i] <= 128'b0;
        end
        else if(wen & ~ren) begin               // write when wen==1 and ren==0
            q <= 128'd0;
            mem[a] <= d;
        end else if(ren & ~wen) begin           // read when ren==1 and wen==0
            q <= mem[a];
        end else begin                          // idle
            q <= 128'd0;
        end
    end

endmodule