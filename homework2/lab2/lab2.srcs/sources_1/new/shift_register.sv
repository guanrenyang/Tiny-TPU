`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/06/2022 11:11:27 PM
// Design Name: 
// Module Name: shift_register
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module shift_register #(parameter MSB=128)(
    input i_clk,
    input i_load,
    input i_dir,
    input [MSB-1:0] din,
    output [31:0] dout
    );
    reg [MSB-1:0] q_mid = 0;
    always@(posedge i_clk) begin
        if(i_load) begin
            q_mid <= din;
        end
        else begin
            case(i_dir)
            1'b0: begin
                q_mid <= {q_mid[MSB-32:0], 32'b0};
            end
            1'b1: begin
                q_mid <= {32'b0, q_mid[MSB-1:32]};
            end
        endcase
        end
    end 
    assign dout = i_dir ? (q_mid[31:0]) : (q_mid[MSB-1:MSB-32]);

endmodule
