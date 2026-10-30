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
