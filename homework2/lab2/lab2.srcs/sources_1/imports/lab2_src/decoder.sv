module decoder
(
    input  wire  [27:0] d,
    input  wire         clk,
    input  wire         reset,
    input  wire         en,

    output [3:0]    func,
    output [9:0]    rs1,
    output [9:0]    rs2,
    output [3:0]    opcode
    );

    reg [3:0]    reg_func;
    reg [9:0]    reg_rs1;
    reg [9:0]    reg_rs2;
    reg [3:0]    reg_opcode;



    always @(posedge clk) begin
        if(~reset)begin
            reg_func <= 4'b0;
            reg_rs1  <= 10'b0;
            reg_rs2  <= 10'b0;
            reg_opcode <= 4'b0;
        end
        else begin
            if (en) begin
                reg_func <= d[27:24];
                reg_rs1  <= d[23:14];
                reg_rs2  <= d[13:4];
                reg_opcode <= d[3:0];
            end
            else begin
                reg_func <= reg_func;
                reg_rs1  <= reg_rs1;
                reg_rs2  <= reg_rs2;
                reg_opcode <= reg_opcode;
            end
        end
    end


    assign func = reg_func;
    assign rs1  = reg_rs1;
    assign rs2  = reg_rs2;
    assign opcode = reg_opcode;


endmodule