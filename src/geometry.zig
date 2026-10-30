const std = @import("std");

pub fn Geometry(comptime N: u16) type {
    return struct {
        pub const SIZE: u16 = N;
        pub const AREA: u16 = N * N;
        pub const SENTINEL: u16 = AREA;
        pub const CONV_NEIGHBORS: [AREA][9]u16 = computeNeighbors();

        fn computeNeighbors() [AREA][9]u16 {
            @setEvalBranchQuota(@as(u32, AREA) * 9 * 16);
            var table: [AREA][9]u16 = undefined;
            for (0..N) |row| {
                for (0..N) |column| {
                    var slot: usize = 0;
                    for (0..3) |kernel_row| {
                        for (0..3) |kernel_column| {
                            const neighbor_row = @as(i32, @intCast(row)) + @as(i32, @intCast(kernel_row)) - 1;
                            const neighbor_column = @as(i32, @intCast(column)) + @as(i32, @intCast(kernel_column)) - 1;
                            const inside = neighbor_row >= 0 and neighbor_row < N and
                                neighbor_column >= 0 and neighbor_column < N;
                            table[row * N + column][slot] = if (inside)
                                @intCast(neighbor_row * N + neighbor_column)
                            else
                                SENTINEL;
                            slot += 1;
                        }
                    }
                }
            }
            return table;
        }
    };
}

const G9 = Geometry(9);
const S = G9.SENTINEL;

test "Geometry(9) sizes" {
    try std.testing.expectEqual(@as(u16, 81), G9.AREA);
    try std.testing.expectEqual(@as(u16, 81), G9.SENTINEL);
}

test "Geometry(9) top-left corner" {
    try std.testing.expectEqual([9]u16{ S, S, S, S, 0, 1, S, 9, 10 }, G9.CONV_NEIGHBORS[0]);
}

test "Geometry(9) bottom-right corner" {
    try std.testing.expectEqual([9]u16{ 70, 71, S, 79, 80, S, S, S, S }, G9.CONV_NEIGHBORS[80]);
}

test "Geometry(9) top-right and bottom-left corners" {
    try std.testing.expectEqual([9]u16{ S, S, S, 7, 8, S, 16, 17, S }, G9.CONV_NEIGHBORS[8]);
    try std.testing.expectEqual([9]u16{ S, 63, 64, S, 72, 73, S, S, S }, G9.CONV_NEIGHBORS[72]);
}

test "Geometry(9) edges" {
    try std.testing.expectEqual([9]u16{ S, S, S, 3, 4, 5, 12, 13, 14 }, G9.CONV_NEIGHBORS[4]);
    try std.testing.expectEqual([9]u16{ S, 27, 28, S, 36, 37, S, 45, 46 }, G9.CONV_NEIGHBORS[36]);
}

test "Geometry(9) center" {
    try std.testing.expectEqual([9]u16{ 30, 31, 32, 39, 40, 41, 48, 49, 50 }, G9.CONV_NEIGHBORS[40]);
}

test "Geometry(9) every interior point has nine on-board neighbors" {
    for (1..8) |row| {
        for (1..8) |column| {
            for (G9.CONV_NEIGHBORS[row * 9 + column]) |neighbor| {
                try std.testing.expect(neighbor < G9.AREA);
            }
        }
    }
}

const G19 = Geometry(19);
const S19 = G19.SENTINEL;

test "Geometry(19) sizes" {
    try std.testing.expectEqual(@as(u16, 361), G19.AREA);
    try std.testing.expectEqual(@as(u16, 361), G19.SENTINEL);
}

test "Geometry(19) corners" {
    try std.testing.expectEqual([9]u16{ S19, S19, S19, S19, 0, 1, S19, 19, 20 }, G19.CONV_NEIGHBORS[0]);
    try std.testing.expectEqual([9]u16{ S19, S19, S19, 17, 18, S19, 36, 37, S19 }, G19.CONV_NEIGHBORS[18]);
    try std.testing.expectEqual([9]u16{ S19, 323, 324, S19, 342, 343, S19, S19, S19 }, G19.CONV_NEIGHBORS[342]);
    try std.testing.expectEqual([9]u16{ 340, 341, S19, 359, 360, S19, S19, S19, S19 }, G19.CONV_NEIGHBORS[360]);
}

test "Geometry(19) edges" {
    try std.testing.expectEqual([9]u16{ S19, S19, S19, 8, 9, 10, 27, 28, 29 }, G19.CONV_NEIGHBORS[9]);
    try std.testing.expectEqual([9]u16{ S19, 152, 153, S19, 171, 172, S19, 190, 191 }, G19.CONV_NEIGHBORS[171]);
    try std.testing.expectEqual([9]u16{ 169, 170, S19, 188, 189, S19, 207, 208, S19 }, G19.CONV_NEIGHBORS[189]);
    try std.testing.expectEqual([9]u16{ 331, 332, 333, 350, 351, 352, S19, S19, S19 }, G19.CONV_NEIGHBORS[351]);
}

test "Geometry(19) center" {
    try std.testing.expectEqual([9]u16{ 160, 161, 162, 179, 180, 181, 198, 199, 200 }, G19.CONV_NEIGHBORS[180]);
}
