require 'csv'

data = CSV.parse(open(ARGV[0]).read)

keys = []

csv_rows = []

data.each_with_index do |row,i|
	if i == 0
		keys = row
	else
		values = {}
		row.each_with_index{|entry, index|values[keys[index]] = entry}
				
		case values["Embarked"]
		when "Q"
			values["Embarked"] = 0
		when "S"
			values["Embarked"] = 1
		when "C"
			values["Embarked"] = 2
		end

		case values["Sex"]
		when "male"
			values["Sex"] = 0
		when "female"
			values["Sex"] = 1
		end

		if values["Age"] == nil
			values["Age"] = -7.3608 * values["Pclass"].to_f + -3.4654 * values["Sex"].to_f + -4.1242 * values["SibSp"].to_f + -0.0166 * values["Fare"].to_f + -2.3978 * values["Embarked"].to_f + 52.86  
		end

		["Cabin", "Ticket", "Name"].each do |key|
			values.delete(key)
		end

		csv_rows << values.values.to_csv
	end
end

["Cabin", "Ticket", "Name"].each do |key|
	keys.delete(key)
end

csv_rows = [keys.to_csv] + csv_rows 

File.open(ARGV[1], "w"){|f| f << csv_rows.join}