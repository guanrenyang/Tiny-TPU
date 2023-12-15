module PE_array#(parameter num = 4)
(
	// interface to system
    input wire clk,
    input wire reset,
    input wire c_en,                              // compute enable
    input wire p_en,                              // preload enable
    input wire e_en,                              // elementwise enable
    // input wire pre_load_cnt,
    // input wire 
    // interface to PE row .....

    input wire signed[31:0]in_weight[num-1:0],      // wire from weight buffer direction
    input wire signed[31:0]in_input[num-1:0],       // wire from input buffer direction
                 
    output reg signed[31:0]result[num-1:0],

    output reg compute_finished

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

    
    // row shift register--shift left
    reg signed [31:0] shift_reg_row_0 [6:0];
    reg signed [31:0] shift_reg_row_1 [6:0];
    reg signed [31:0] shift_reg_row_2 [6:0];
    reg signed [31:0] shift_reg_row_3 [6:0];

    // coloumn shift register--
    reg signed [31:0] shift_reg_col_0 [6:0];
    reg signed [31:0] shift_reg_col_1 [6:0];
    reg signed [31:0] shift_reg_col_2 [6:0];
    reg signed [31:0] shift_reg_col_3 [6:0];

    //output
    reg signed [31:0] result_row_0 [3:0];
    reg signed [31:0] result_row_1 [3:0];
    reg signed [31:0] result_row_2 [3:0];
    reg signed [31:0] result_row_3 [3:0];


    reg [3:0] comp_cnt;
    reg [2:0] preload_cnt;
    reg [2:0] write_cnt; // for elementwise array load
    integer i;
    always @(posedge clk) begin
        if(~reset) begin
            // do some intialization
            comp_cnt <= 0;
            compute_finished <= 0;
            write_cnt<=0;
            for(i=0;i<7;i++) begin
                shift_reg_row_0[i]<=0;
                shift_reg_row_1[i]<=0;
                shift_reg_row_2[i]<=0;
                shift_reg_row_3[i]<=0;
            end
            for(i=0;i<7;i++) begin
                shift_reg_col_0[i]<=0;
                shift_reg_col_1[i]<=0;
                shift_reg_col_2[i]<=0;
                shift_reg_col_3[i]<=0;
            end
            //preload
            preload_cnt <= 0;
        end else if (c_en&&~p_en&&~e_en) begin // 给移位寄存器赋值
            comp_cnt <= comp_cnt+1;

            if (comp_cnt>=1&&comp_cnt<=4)begin 
                //shift
                //load weight
                shift_reg_col_0[0] <= in_weight[3];
                shift_reg_col_1[0] <= in_weight[2];
                shift_reg_col_2[0] <= in_weight[1];
                shift_reg_col_3[0] <= in_weight[0];
                
                {shift_reg_col_0[3],shift_reg_col_0[2],shift_reg_col_0[1]} <= {shift_reg_col_0[2],shift_reg_col_0[1], shift_reg_col_0[0]};
                {shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1]} <= { shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1], shift_reg_col_1[0]};
                {shift_reg_col_2[5], shift_reg_col_2[4], shift_reg_col_2[3],shift_reg_col_2[2], shift_reg_col_2[1]} <= {shift_reg_col_2[4], shift_reg_col_2[3],shift_reg_col_2[2], shift_reg_col_2[1], shift_reg_col_2[0]};
                {shift_reg_col_3[6], shift_reg_col_3[5], shift_reg_col_3[4],shift_reg_col_3[3], shift_reg_col_3[2], shift_reg_col_3[1]} <= {shift_reg_col_3[5], shift_reg_col_3[4], shift_reg_col_3[3],shift_reg_col_3[2], shift_reg_col_3[1], shift_reg_col_3[0]};
                case (comp_cnt)
                    1: begin
                        shift_reg_row_0[3] <= in_input[3];
                        shift_reg_row_0[2] <= in_input[2];
                        shift_reg_row_0[1] <= in_input[1];
                        shift_reg_row_0[0] <= in_input[0];                                                
                    end
                    2: begin
                        shift_reg_row_1[3] <= in_input[3];
                        shift_reg_row_1[2] <= in_input[2];
                        shift_reg_row_1[1] <= in_input[1];
                        shift_reg_row_1[0] <= in_input[0];

                        //shift
                        {shift_reg_row_0[6],shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0]} <= 
                            {shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0], 32'b0};

                        // {shift_reg_col_0[3],shift_reg_col_0[2],shift_reg_col_0[1]} <= {shift_reg_col_0[2],shift_reg_col_0[1], shift_reg_col_0[0]};
                    end
                    3: begin
                        shift_reg_row_2[3] <= in_input[3];
                        shift_reg_row_2[2] <= in_input[2];
                        shift_reg_row_2[1] <= in_input[1];
                        shift_reg_row_2[0] <= in_input[0];
                        
                        //shift
                        {shift_reg_row_0[6],shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0]} <= 
                            {shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0], 32'b0};
                        {shift_reg_row_1[6],shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0]} <= 
                            {shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0], 32'b0};

                        // {shift_reg_col_0[3],shift_reg_col_0[2],shift_reg_col_0[1]} <= {shift_reg_col_0[2],shift_reg_col_0[1], shift_reg_col_0[0]};
                        // {shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1]} <= { shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1], shift_reg_col_1[0]};
                    end
                    4: begin
                        shift_reg_row_3[3] <= in_input[3];
                        shift_reg_row_3[2] <= in_input[2];
                        shift_reg_row_3[1] <= in_input[1];
                        shift_reg_row_3[0] <= in_input[0];

                        //shift
                        {shift_reg_row_0[6],shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0]} <= 
                            {shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0], 32'b0};
                        {shift_reg_row_1[6],shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0]} <= 
                            {shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0], 32'b0};
                        {shift_reg_row_2[6],shift_reg_row_2[5],shift_reg_row_2[4],shift_reg_row_2[3],shift_reg_row_2[2],shift_reg_row_2[1],shift_reg_row_2[0]} <= 
                            {shift_reg_row_2[5],shift_reg_row_2[4],shift_reg_row_2[3],shift_reg_row_2[2],shift_reg_row_2[1],shift_reg_row_2[0], 32'b0};

                        // {shift_reg_col_0[3],shift_reg_col_0[2],shift_reg_col_0[1]} <= {shift_reg_col_0[2],shift_reg_col_0[1], shift_reg_col_0[0]};
                        // {shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1]} <= { shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1], shift_reg_col_1[0]};
                        // {shift_reg_col_1[5], shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1]} <= {shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1], shift_reg_col_1[0]};
                    end
                endcase
            end else if (comp_cnt>4&& comp_cnt<=11) begin
                //shift
                {shift_reg_row_0[6],shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0]} <= 
                    {shift_reg_row_0[5],shift_reg_row_0[4],shift_reg_row_0[3],shift_reg_row_0[2],shift_reg_row_0[1],shift_reg_row_0[0], 32'b0};
                {shift_reg_row_1[6],shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0]} <= 
                    {shift_reg_row_1[5],shift_reg_row_1[4],shift_reg_row_1[3],shift_reg_row_1[2],shift_reg_row_1[1],shift_reg_row_1[0], 32'b0};
                {shift_reg_row_2[6],shift_reg_row_2[5],shift_reg_row_2[4],shift_reg_row_2[3],shift_reg_row_2[2],shift_reg_row_2[1],shift_reg_row_2[0]} <= 
                    {shift_reg_row_2[5],shift_reg_row_2[4],shift_reg_row_2[3],shift_reg_row_2[2],shift_reg_row_2[1],shift_reg_row_2[0], 32'b0};
                {shift_reg_row_3[6],shift_reg_row_3[5],shift_reg_row_3[4],shift_reg_row_3[3],shift_reg_row_3[2],shift_reg_row_3[1],shift_reg_row_3[0]} <= 
                    {shift_reg_row_3[5],shift_reg_row_3[4],shift_reg_row_3[3],shift_reg_row_3[2],shift_reg_row_3[1],shift_reg_row_3[0], 32'b0};

                {shift_reg_col_0[3],shift_reg_col_0[2],shift_reg_col_0[1]} <= {shift_reg_col_0[2],shift_reg_col_0[1], shift_reg_col_0[0]};
                {shift_reg_col_1[4], shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1]} <= { shift_reg_col_1[3],shift_reg_col_1[2], shift_reg_col_1[1], shift_reg_col_1[0]};
                {shift_reg_col_2[5], shift_reg_col_2[4], shift_reg_col_2[3],shift_reg_col_2[2], shift_reg_col_2[1]} <= {shift_reg_col_2[4], shift_reg_col_2[3],shift_reg_col_2[2], shift_reg_col_2[1], shift_reg_col_2[0]};
                {shift_reg_col_3[6], shift_reg_col_3[5], shift_reg_col_3[4],shift_reg_col_3[3], shift_reg_col_3[2], shift_reg_col_3[1]} <= {shift_reg_col_3[5], shift_reg_col_3[4], shift_reg_col_3[3],shift_reg_col_3[2], shift_reg_col_3[1], shift_reg_col_3[0]};
                if(comp_cnt==11)
                    compute_finished<=1;
            end

        end else if (~c_en&&p_en&&~e_en) begin
            // do preload
            preload_cnt <= preload_cnt+1;
        end else if(~c_en&&~p_en&&e_en)begin
            write_cnt<=write_cnt+1;
        end else begin
            write_cnt<=0;
            comp_cnt <= 0;
            compute_finished <= 0;
            preload_cnt<=0;
            for(i=0;i<7;i++) begin
                shift_reg_row_0[i]<=0;
                shift_reg_row_1[i]<=0;
                shift_reg_row_2[i]<=0;
                shift_reg_row_3[i]<=0;
            end
            for(i=0;i<7;i++) begin
                shift_reg_col_0[i]<=0;
                shift_reg_col_1[i]<=0;
                shift_reg_col_2[i]<=0;
                shift_reg_col_3[i]<=0;
            end
            // PE array doesn't work
            // comp_cnt <= 0;
            // integer  i=0;
            // for(i=0;i<7;i++) begin
        
        end
    end
   
    always @(*) begin
        if(p_en&&~c_en)
        case(preload_cnt) // 从input口进行preload
                0: begin
                    shift_reg_col_0[0] <= in_weight[3];
                    shift_reg_col_1[1] <= in_weight[2];
                    shift_reg_col_2[2] <= in_weight[1];
                    shift_reg_col_3[3] <= in_weight[0];
                end
                1: begin                               
                    shift_reg_col_0[1] <= in_weight[3];
                    shift_reg_col_1[2] <= in_weight[2];
                    shift_reg_col_2[3] <= in_weight[1];
                    shift_reg_col_3[4] <= in_weight[0];
                end
                2: begin                               
                    shift_reg_col_0[2] <= in_weight[3];
                    shift_reg_col_1[3] <= in_weight[2];
                    shift_reg_col_2[4] <= in_weight[1];
                    shift_reg_col_3[5] <= in_weight[0];
                end
                3: begin                               
                    shift_reg_col_0[3] <= in_weight[3];
                    shift_reg_col_1[4] <= in_weight[2];
                    shift_reg_col_2[5] <= in_weight[1];
                    shift_reg_col_3[6] <= in_weight[0];
                end
            endcase
    end
    // to do: insert buffers in front of different rows or columns of PE array to ensure logic correction
    always @(*) begin
        if(e_en&&~c_en&&!p_en) begin
            case (write_cnt)
                0: {result[3],result[2],result[1],result[0]} = {result_row_0[3],result_row_0[2],result_row_0[1],result_row_0[0]};
                1: {result[3],result[2],result[1],result[0]} = {result_row_1[3],result_row_1[2],result_row_1[1],result_row_1[0]};
                2: {result[3],result[2],result[1],result[0]} = {result_row_2[3],result_row_2[2],result_row_2[1],result_row_2[0]};
                3: {result[3],result[2],result[1],result[0]} = {result_row_3[3],result_row_3[2],result_row_3[1],result_row_3[0]};
            endcase
        end
    end


    // to do: some glue jobs

    PE PE_00(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_0[0]),
        .in_input(shift_reg_row_0[3]),
        .result(result_row_0[3])
    );
    PE PE_01(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_1[1]),
        .in_input(shift_reg_row_0[4]),
        .result(result_row_0[2])
    );
    PE PE_02(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_2[2]),
        .in_input(shift_reg_row_0[5]),
        .result(result_row_0[1])
    );
    PE PE_03(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_3[3]),
        .in_input(shift_reg_row_0[6]),
        .result(result_row_0[0])
    );

    PE PE_10(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_0[1]),
        .in_input(shift_reg_row_1[3]),
        .result(result_row_1[3])
    );
    PE PE_11(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_1[2]),
        .in_input(shift_reg_row_1[4]),
        .result(result_row_1[2])
    );
    PE PE_12(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_2[3]),
        .in_input(shift_reg_row_1[5]),
        .result(result_row_1[1])
    );
    PE PE_13(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_3[4]),
        .in_input(shift_reg_row_1[6]),
        .result(result_row_1[0])
    );
    PE PE_20(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_0[2]),
        .in_input(shift_reg_row_2[3]),
        .result(result_row_2[3])
    );
    PE PE_21(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_1[3]),
        .in_input(shift_reg_row_2[4]),
        .result(result_row_2[2])
    );
    PE PE_22(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_2[4]),
        .in_input(shift_reg_row_2[5]),
        .result(result_row_2[1])
    );
    PE PE_23(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_3[5]),
        .in_input(shift_reg_row_2[6]),
        .result(result_row_2[0])
    );
    PE PE_30(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_0[3]),
        .in_input(shift_reg_row_3[3]),
        .result(result_row_3[3])
    );
    PE PE_31(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_1[4]),
        .in_input(shift_reg_row_3[4]),
        .result(result_row_3[2])
    );
    PE PE_32(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_2[5]),
        .in_input(shift_reg_row_3[5]),
        .result(result_row_3[1])
    );
    PE PE_33(
        .clk(clk),
        .reset(reset),
        .c_en(c_en),
        .p_en(p_en),
        .in_weight(shift_reg_col_3[6]),
        .in_input(shift_reg_row_3[6]),
        .result(result_row_3[0])
    );


endmodule
